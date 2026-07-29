#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <type_traits>
#include <vector>

namespace {

constexpr int64_t kSharedMaxBlockSize = 16384;
constexpr int64_t kGlobalMaxBlockSize = 1LL << 23;

static inline int64_t padded_shared_size(int64_t logical_size) {
  return logical_size + ((logical_size + 31) / 32);
}

static inline int64_t next_power_of_two(int64_t value) {
  int64_t out = 1;
  while (out < value) out <<= 1;
  return out;
}

__device__ __forceinline__ uint32_t lowbias32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

__device__ __forceinline__ uint32_t sign_word_for(uint64_t seed, int64_t block, int layer, int word) {
  uint32_t seed32 = (uint32_t)seed ^ (uint32_t)(seed >> 32);
  uint32_t x = seed32;
  x ^= 0x9e3779b9U * (uint32_t)(block + 1);
  x ^= 0x85ebca6bU * (uint32_t)(layer + 1);
  x ^= 0xc2b2ae35U * (uint32_t)(word + 1);
  return lowbias32(x);
}

__device__ __forceinline__ float sign_for(uint64_t seed, int64_t block, int layer, int pos) {
  uint32_t bits = sign_word_for(seed, block, layer, pos >> 5);
  return ((bits >> (pos & 31)) & 1U) ? 1.0f : -1.0f;
}

__device__ __forceinline__ int smem_index(int pos) {
  return pos + (pos >> 5);
}

template <typename scalar_t>
__device__ __forceinline__ float scalar_to_float(scalar_t value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float scalar_to_float<at::Half>(at::Half value) {
  return __half2float(*reinterpret_cast<const __half*>(&value));
}

template <>
__device__ __forceinline__ float scalar_to_float<at::BFloat16>(at::BFloat16 value) {
  return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&value));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t float_to_scalar(float value) {
  return static_cast<scalar_t>(value);
}

template <>
__device__ __forceinline__ at::Half float_to_scalar<at::Half>(float value) {
  __half out = __float2half(value);
  return *reinterpret_cast<at::Half*>(&out);
}

template <>
__device__ __forceinline__ at::BFloat16 float_to_scalar<at::BFloat16>(float value) {
  __nv_bfloat16 out = __float2bfloat16(value);
  return *reinterpret_cast<at::BFloat16*>(&out);
}

template <typename scalar_t>
__device__ void init_signed_latent(float* values, const scalar_t* latent, int latent_size, int block_size, uint64_t seed, int64_t block) {
  int words = (block_size + 31) >> 5;
  for (int word = threadIdx.x; word < words; word += blockDim.x) {
    uint32_t bits = sign_word_for(seed, block, 0, word);
    #pragma unroll
    for (int bit = 0; bit < 32; ++bit) {
      int pos = (word << 5) + bit;
      if (pos < block_size) {
        float v = pos < latent_size ? scalar_to_float(latent[pos]) : 0.0f;
        values[smem_index(pos)] = v * (((bits >> bit) & 1U) ? 1.0f : -1.0f);
      }
    }
  }
}

__device__ void apply_sign_wordwise(float* values, int block_size, uint64_t seed, int64_t block, int layer) {
  int words = (block_size + 31) >> 5;
  for (int word = threadIdx.x; word < words; word += blockDim.x) {
    uint32_t bits = sign_word_for(seed, block, layer, word);
    #pragma unroll
    for (int bit = 0; bit < 32; ++bit) {
      int pos = (word << 5) + bit;
      if (pos < block_size) {
        values[smem_index(pos)] *= ((bits >> bit) & 1U) ? 1.0f : -1.0f;
      }
    }
  }
}

__device__ __forceinline__ float dot4(float4 a, float4 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ void fht_inplace(float* values, int n) {
  for (int step = 1; step < n; step <<= 1) {
    __syncthreads();
    for (int i = threadIdx.x; i < n / 2; i += blockDim.x) {
      int group = i / step;
      int lane = i - group * step;
      int a = group * (step << 1) + lane;
      int b = a + step;
      int pa = smem_index(a);
      int pb = smem_index(b);
      float x = values[pa];
      float y = values[pb];
      values[pa] = x + y;
      values[pb] = x - y;
    }
  }
  __syncthreads();
  float scale = rsqrtf((float)n);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    values[smem_index(i)] *= scale;
  }
  __syncthreads();
}

__global__ void block_fht_forward_kernel(
    const float* __restrict__ latent,
    float* __restrict__ out,
    int latent_size,
    int output_size,
    int block_size,
    int layers,
    uint64_t seed,
    int start,
    int stop) {
  extern __shared__ float values[];
  int first_block = start / block_size;
  int block = first_block + blockIdx.x;
  int block_start = block * block_size;
  int local_start = max(start - block_start, 0);
  int local_stop = min(stop - block_start, block_size);

  init_signed_latent(values, latent, latent_size, block_size, seed, block);
  __syncthreads();

  for (int layer = 0; layer < layers; ++layer) {
    fht_inplace(values, block_size);
    apply_sign_wordwise(values, block_size, seed, block, layer + 1);
    __syncthreads();
  }

  for (int i = local_start + threadIdx.x; i < local_stop; i += blockDim.x) {
    int global_index = block_start + i;
    if (global_index < output_size) {
      out[global_index - start] = values[smem_index(i)];
    }
  }
}

__global__ void block_fht_backward_kernel(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_latent,
    int latent_size,
    int output_size,
    int block_size,
    int layers,
    uint64_t seed,
    int start,
    int stop) {
  extern __shared__ float values[];
  int first_block = start / block_size;
  int block = first_block + blockIdx.x;
  int block_start = block * block_size;
  int local_start = max(start - block_start, 0);
  int local_stop = min(stop - block_start, block_size);

  for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
    values[smem_index(i)] = 0.0f;
  }
  __syncthreads();
  for (int i = local_start + threadIdx.x; i < local_stop; i += blockDim.x) {
    int global_index = block_start + i;
    if (global_index < output_size) {
      values[smem_index(i)] = grad_out[global_index - start];
    }
  }
  __syncthreads();

  for (int layer = layers - 1; layer >= 0; --layer) {
    apply_sign_wordwise(values, block_size, seed, block, layer + 1);
    __syncthreads();
    fht_inplace(values, block_size);
  }
  for (int i = threadIdx.x; i < latent_size; i += blockDim.x) {
    float g = values[smem_index(i)] * sign_for(seed, block, 0, i);
    atomicAdd(grad_latent + i, g);
  }
}

template <typename scalar_t>
__global__ __launch_bounds__(256) void block_fht_linear_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ latent,
    scalar_t* __restrict__ out,
    int tokens,
    int in_features,
    int out_features,
    int latent_size,
    int block_size,
    int layers,
    uint64_t seed,
    float weight_scale) {
  extern __shared__ float values[];
  constexpr int kTokenTile = 8;
  int block = blockIdx.x;
  int token_start = blockIdx.y * kTokenTile;
  int output_size = in_features * out_features;
  int block_start = block * block_size;
  int local_stop = min(block_size, output_size - block_start);
  int rows_per_block = block_size / in_features;
  int out_row_start = block * rows_per_block;
  int active_rows = min(rows_per_block, out_features - out_row_start);

  init_signed_latent(values, latent, latent_size, block_size, seed, block);
  __syncthreads();

  for (int layer = 0; layer < layers; ++layer) {
    fht_inplace(values, block_size);
    apply_sign_wordwise(values, block_size, seed, block, layer + 1);
    __syncthreads();
  }

  if (weight_scale != 1.0f) {
    for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
      values[smem_index(i)] *= weight_scale;
    }
    __syncthreads();
  }

  if (local_stop <= 0 || active_rows <= 0) {
    return;
  }

  // One CTA owns a complete row-block for a small token tile. No atomics:
  // each warp reduces one output element, so rows/token pairs run concurrently.
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  constexpr int kWarpsPerCTA = 256 / 32;
  for (int pair = warp; pair < kTokenTile * active_rows; pair += kWarpsPerCTA) {
    int token_offset = pair / active_rows;
    int local_row = pair - token_offset * active_rows;
    int token = token_start + token_offset;
    if (token >= tokens) {
      continue;
    }
    float partial = 0.0f;
    int row_base = local_row * in_features;
    if constexpr (std::is_same<scalar_t, float>::value) {
      if ((in_features & 3) == 0) {
      const float4* input4 = reinterpret_cast<const float4*>(input + token * in_features);
      int in_features4 = in_features >> 2;
      for (int i = lane; i < in_features4; i += 32) {
        int base = row_base + (i << 2);
        float4 w = make_float4(
            values[smem_index(base)],
            values[smem_index(base + 1)],
            values[smem_index(base + 2)],
            values[smem_index(base + 3)]);
        partial += dot4(input4[i], w);
      }
      } else {
        for (int i = lane; i < in_features; i += 32) {
          partial += input[token * in_features + i] * values[smem_index(row_base + i)];
        }
      }
    } else if constexpr (std::is_same<scalar_t, at::Half>::value) {
      if ((in_features & 1) == 0) {
        const __half2* input2 = reinterpret_cast<const __half2*>(input + token * in_features);
        int in_features2 = in_features >> 1;
        for (int i = lane; i < in_features2; i += 32) {
          float2 x = __half22float2(input2[i]);
          int base = row_base + (i << 1);
          partial += x.x * values[smem_index(base)] + x.y * values[smem_index(base + 1)];
        }
      } else {
        for (int i = lane; i < in_features; i += 32) {
          partial += static_cast<float>(input[token * in_features + i]) * values[smem_index(row_base + i)];
        }
      }
    } else if constexpr (std::is_same<scalar_t, at::BFloat16>::value) {
      for (int i = lane; i < in_features; i += 32) {
        partial += scalar_to_float(input[token * in_features + i]) * values[smem_index(row_base + i)];
      }
    } else {
      for (int i = lane; i < in_features; i += 32) {
        partial += scalar_to_float(input[token * in_features + i]) * values[smem_index(row_base + i)];
      }
    }
    float sum = warp_sum(partial);
    if (lane == 0) {
      out[token * out_features + out_row_start + local_row] = float_to_scalar<scalar_t>(sum);
    }
  }
}

template <typename scalar_t>
__global__ __launch_bounds__(256) void fixed_basis_transform_kernel(
    const scalar_t* __restrict__ input,
    const int64_t* __restrict__ permutations,
    const float* __restrict__ signs,
    scalar_t* __restrict__ output,
    int64_t tokens,
    int64_t width,
    int64_t block_size,
    int64_t bases,
    int64_t jobs,
    bool inverse,
    bool shared_input) {
  extern __shared__ float values[];
  int64_t blocks_per_token = width / block_size;
  for (int64_t job = blockIdx.x; job < jobs; job += gridDim.x) {
    int64_t basis = job / (tokens * blocks_per_token);
    int64_t residual = job - basis * tokens * blocks_per_token;
    int64_t token = residual / blocks_per_token;
    int64_t transform_block = residual - token * blocks_per_token;
    int64_t spectral_start = transform_block * block_size;
    int64_t basis_offset = basis * width;
    int64_t input_token_offset =
        (shared_input ? token : basis * tokens + token) * width;
    int64_t output_token_offset = (basis * tokens + token) * width;

    for (int64_t lane = threadIdx.x; lane < block_size;
         lane += blockDim.x) {
      int64_t spectral_index = spectral_start + lane;
      int64_t permutation = permutations[basis_offset + spectral_index];
      int64_t input_index = inverse ? spectral_index : permutation;
      float value = scalar_to_float(input[input_token_offset + input_index]);
      if (inverse) {
        value *= signs[basis_offset + spectral_index];
      }
      values[smem_index((int)lane)] = value;
    }
    __syncthreads();

    fht_inplace(values, (int)block_size);

    for (int64_t lane = threadIdx.x; lane < block_size;
         lane += blockDim.x) {
      int64_t spectral_index = spectral_start + lane;
      int64_t permutation = permutations[basis_offset + spectral_index];
      int64_t output_index = inverse ? permutation : spectral_index;
      float value = values[smem_index((int)lane)];
      if (!inverse) {
        value *= signs[basis_offset + spectral_index];
      }
      output[output_token_offset + output_index] =
          float_to_scalar<scalar_t>(value);
    }
    __syncthreads();
  }
}

template <typename scalar_t>
__global__ __launch_bounds__(64) void fixed_basis_transform_256_kernel(
    const scalar_t* __restrict__ input,
    const int64_t* __restrict__ permutations,
    const float* __restrict__ signs,
    scalar_t* __restrict__ output,
    int64_t tokens,
    int64_t width,
    int64_t bases,
    int64_t jobs,
    bool inverse,
    bool shared_input) {
  extern __shared__ float values[];
  constexpr int64_t block_size = 256;
  int64_t blocks_per_token = width / block_size;
  int warp = (int)(threadIdx.x >> 5);
  int warp_lane = (int)(threadIdx.x & 31);
  for (int64_t job = blockIdx.x; job < jobs; job += gridDim.x) {
    int64_t basis = job / (tokens * blocks_per_token);
    int64_t residual = job - basis * tokens * blocks_per_token;
    int64_t token = residual / blocks_per_token;
    int64_t transform_block = residual - token * blocks_per_token;
    int64_t basis_offset = basis * width;
    int64_t input_token_offset =
        (shared_input ? token : basis * tokens + token) * width;
    int64_t spectral_indices[4];
    int64_t selected_permutations[4];
    #pragma unroll
    for (int quarter = 0; quarter < 4; ++quarter) {
      int group = warp + quarter * 2;
      int64_t spectral_index =
          transform_block * block_size + group * 32 + warp_lane;
      int64_t permutation = permutations[basis_offset + spectral_index];
      int64_t input_index = inverse ? spectral_index : permutation;
      float value =
          scalar_to_float(input[input_token_offset + input_index]);
      if (inverse) {
        value *= signs[basis_offset + spectral_index];
      }
      #pragma unroll
      for (int offset = 1; offset <= 16; offset <<= 1) {
        float partner = __shfl_xor_sync(0xffffffff, value, offset);
        value =
            (warp_lane & offset) ? partner - value : value + partner;
      }
      spectral_indices[quarter] = spectral_index;
      selected_permutations[quarter] = permutation;
      values[group * 33 + warp_lane] = value;
    }
    __syncthreads();

    // H256 = H8 (across warps) kron H32 (within each warp). Each thread
    // computes four H8 rows directly from the eight shared warp values.
    int64_t output_token_offset = (basis * tokens + token) * width;
    #pragma unroll
    for (int quarter = 0; quarter < 4; ++quarter) {
      int output_group = warp + quarter * 2;
      float cross_warp = 0.0f;
      #pragma unroll
      for (int input_group = 0; input_group < 8; ++input_group) {
        float component = values[input_group * 33 + warp_lane];
        cross_warp += (__popc(output_group & input_group) & 1)
            ? -component
            : component;
      }
      float value = cross_warp * 0.0625f;
      int64_t spectral_index = spectral_indices[quarter];
      int64_t output_index =
          inverse ? selected_permutations[quarter] : spectral_index;
      if (!inverse) {
        value *= signs[basis_offset + spectral_index];
      }
      output[output_token_offset + output_index] =
          float_to_scalar<scalar_t>(value);
    }
    // The persistent CTA reuses the same shared rows for its next job.
    __syncthreads();
  }
}

template <typename scalar_t>
__global__ void postgelu_mix_forward_kernel(
    const scalar_t* __restrict__ update,
    const scalar_t* __restrict__ condition,
    const float* __restrict__ slope,
    scalar_t* __restrict__ correction,
    int64_t elements,
    int64_t tokens,
    int64_t width,
    float scale) {
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements;
       index += (int64_t)blockDim.x * gridDim.x) {
    int64_t spectral = index % width;
    int64_t basis = index / (tokens * width);
    float update_value = scalar_to_float(update[index]);
    float condition_value = scalar_to_float(condition[index]);
    float slope_value =
        scalar_to_float(float_to_scalar<scalar_t>(
            slope[basis * width + spectral]));
    float first = scalar_to_float(
        float_to_scalar<scalar_t>(update_value * condition_value));
    correction[index] =
        float_to_scalar<scalar_t>(scale * first * slope_value);
  }
}

template <typename scalar_t>
__global__ void postgelu_mix_backward_kernel(
    const scalar_t* __restrict__ grad_correction,
    const scalar_t* __restrict__ update,
    const scalar_t* __restrict__ condition,
    const float* __restrict__ slope,
    scalar_t* __restrict__ grad_update,
    scalar_t* __restrict__ grad_condition,
    int64_t elements,
    int64_t tokens,
    int64_t width,
    float scale) {
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements;
       index += (int64_t)blockDim.x * gridDim.x) {
    int64_t spectral = index % width;
    int64_t basis = index / (tokens * width);
    float grad_value = scalar_to_float(grad_correction[index]);
    float update_value = scalar_to_float(update[index]);
    float condition_value = scalar_to_float(condition[index]);
    float slope_value =
        scalar_to_float(float_to_scalar<scalar_t>(
            slope[basis * width + spectral]));
    grad_update[index] = float_to_scalar<scalar_t>(
        scale * grad_value * condition_value * slope_value);
    grad_condition[index] = float_to_scalar<scalar_t>(
        scale * grad_value * update_value * slope_value);
  }
}

template <typename scalar_t>
__global__ void postgelu_slope_backward_kernel(
    const scalar_t* __restrict__ grad_correction,
    const scalar_t* __restrict__ update,
    const scalar_t* __restrict__ condition,
    float* __restrict__ grad_slope,
    int64_t bases,
    int64_t tokens,
    int64_t width,
    int64_t token_chunks,
    float scale) {
  int64_t width_blocks = (width + blockDim.x - 1) / blockDim.x;
  int64_t basis = blockIdx.x / (width_blocks * token_chunks);
  int64_t residual =
      blockIdx.x - basis * width_blocks * token_chunks;
  int64_t width_block = residual / token_chunks;
  int64_t token_chunk = residual - width_block * token_chunks;
  int64_t spectral = width_block * blockDim.x + threadIdx.x;
  if (basis >= bases || spectral >= width) {
    return;
  }
  int64_t token_start = tokens * token_chunk / token_chunks;
  int64_t token_stop = tokens * (token_chunk + 1) / token_chunks;
  int64_t basis_offset = basis * tokens * width;
  float total = 0.0f;
  for (int64_t token = token_start; token < token_stop; ++token) {
    int64_t index = basis_offset + token * width + spectral;
    float grad_value = scalar_to_float(grad_correction[index]);
    float update_value = scalar_to_float(update[index]);
    float condition_value = scalar_to_float(condition[index]);
    float first = scalar_to_float(
        float_to_scalar<scalar_t>(grad_value * update_value));
    total += scalar_to_float(
        float_to_scalar<scalar_t>(first * condition_value));
  }
  atomicAdd(grad_slope + basis * width + spectral, scale * total);
}

template <typename scalar_t>
__global__ void postgelu_sum_heads_kernel(
    const scalar_t* __restrict__ per_head,
    const scalar_t* __restrict__ residual,
    scalar_t* __restrict__ output,
    int64_t bases,
    int64_t tokens,
    int64_t width,
    bool add_residual) {
  int64_t elements = tokens * width;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements;
       index += (int64_t)blockDim.x * gridDim.x) {
    float total = add_residual ? scalar_to_float(residual[index]) : 0.0f;
    for (int64_t basis = 0; basis < bases; ++basis) {
      total += scalar_to_float(per_head[basis * elements + index]);
    }
    output[index] = float_to_scalar<scalar_t>(total);
  }
}

__global__ void init_forward_blocks_kernel(
    const float* __restrict__ latent,
    float* __restrict__ work,
    int64_t latent_size,
    int64_t block_size,
    int64_t first_block,
    int64_t block_count,
    uint64_t seed) {
  int64_t total = block_count * block_size;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total;
       index += (int64_t)blockDim.x * gridDim.x) {
    int64_t local_block = index / block_size;
    int64_t pos = index - local_block * block_size;
    int64_t block = first_block + local_block;
    float value = pos < latent_size ? latent[pos] : 0.0f;
    work[index] = value * sign_for(seed, block, 0, (int)pos);
  }
}

__global__ void init_backward_blocks_kernel(
    const float* __restrict__ grad_out,
    float* __restrict__ work,
    int64_t output_size,
    int64_t block_size,
    int64_t first_block,
    int64_t block_count,
    int64_t start,
    int64_t stop) {
  int64_t total = block_count * block_size;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total;
       index += (int64_t)blockDim.x * gridDim.x) {
    int64_t local_block = index / block_size;
    int64_t pos = index - local_block * block_size;
    int64_t global_index = (first_block + local_block) * block_size + pos;
    float value = 0.0f;
    if (global_index >= start && global_index < stop && global_index < output_size) {
      value = grad_out[global_index - start];
    }
    work[index] = value;
  }
}

__global__ void fht_step_blocks_kernel(
    float* __restrict__ work,
    int64_t block_size,
    int64_t pair_count,
    int64_t step) {
  for (int64_t pair = blockIdx.x * blockDim.x + threadIdx.x; pair < pair_count;
       pair += (int64_t)blockDim.x * gridDim.x) {
    int64_t local_pair = pair % (block_size / 2);
    int64_t block = pair / (block_size / 2);
    int64_t group = local_pair / step;
    int64_t lane = local_pair - group * step;
    int64_t a = block * block_size + group * (step << 1) + lane;
    int64_t b = a + step;
    float x = work[a];
    float y = work[b];
    work[a] = x + y;
    work[b] = x - y;
  }
}

__global__ void scale_blocks_kernel(
    float* __restrict__ work,
    int64_t total,
    float scale) {
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total;
       index += (int64_t)blockDim.x * gridDim.x) {
    work[index] *= scale;
  }
}

__global__ void apply_sign_blocks_kernel(
    float* __restrict__ work,
    int64_t block_size,
    int64_t first_block,
    int64_t block_count,
    int layer,
    uint64_t seed) {
  int64_t total = block_count * block_size;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total;
       index += (int64_t)blockDim.x * gridDim.x) {
    int64_t local_block = index / block_size;
    int64_t pos = index - local_block * block_size;
    int64_t block = first_block + local_block;
    work[index] *= sign_for(seed, block, layer, (int)pos);
  }
}

__global__ void apply_sign_scale_blocks_kernel(
    float* __restrict__ work,
    int64_t block_size,
    int64_t first_block,
    int64_t block_count,
    int layer,
    uint64_t seed,
    float scale) {
  int64_t total = block_count * block_size;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total;
       index += (int64_t)blockDim.x * gridDim.x) {
    int64_t local_block = index / block_size;
    int64_t pos = index - local_block * block_size;
    int64_t block = first_block + local_block;
    work[index] *= scale * sign_for(seed, block, layer, (int)pos);
  }
}

__global__ void gather_slice_kernel(
    const float* __restrict__ work,
    float* __restrict__ out,
    int64_t output_size,
    int64_t block_size,
    int64_t first_block,
    int64_t start,
    int64_t stop) {
  int64_t length = stop - start;
  for (int64_t offset = blockIdx.x * blockDim.x + threadIdx.x; offset < length;
       offset += (int64_t)blockDim.x * gridDim.x) {
    int64_t global_index = start + offset;
    if (global_index < output_size) {
      int64_t block = global_index / block_size;
      int64_t pos = global_index - block * block_size;
      out[offset] = work[(block - first_block) * block_size + pos];
    }
  }
}

__global__ void accumulate_latent_grad_kernel(
    const float* __restrict__ work,
    float* __restrict__ grad_latent,
    int64_t latent_size,
    int64_t block_size,
    int64_t first_block,
    int64_t block_count,
    uint64_t seed) {
  int64_t total = block_count * latent_size;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total;
       index += (int64_t)blockDim.x * gridDim.x) {
    int64_t local_block = index / latent_size;
    int64_t pos = index - local_block * latent_size;
    int64_t block = first_block + local_block;
    float value = work[local_block * block_size + pos] * sign_for(seed, block, 0, (int)pos);
    atomicAdd(grad_latent + pos, value);
  }
}

__global__ void accumulate_latent_grad_scaled_kernel(
    const float* __restrict__ work,
    float* __restrict__ grad_latent,
    int64_t latent_size,
    int64_t block_size,
    int64_t first_block,
    int64_t block_count,
    uint64_t seed,
    float scale) {
  int64_t total = block_count * latent_size;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total;
       index += (int64_t)blockDim.x * gridDim.x) {
    int64_t local_block = index / latent_size;
    int64_t pos = index - local_block * latent_size;
    int64_t block = first_block + local_block;
    float value = scale * work[local_block * block_size + pos] * sign_for(seed, block, 0, (int)pos);
    atomicAdd(grad_latent + pos, value);
  }
}

int launch_blocks_for(int64_t work_items) {
  int64_t blocks = (work_items + 255) / 256;
  if (blocks < 1) return 1;
  if (blocks > 65535) return 65535;
  return (int)blocks;
}

void run_global_fht_unscaled(torch::Tensor work, int64_t block_size, int64_t block_count, cudaStream_t stream) {
  int threads = 256;
  int64_t pair_count = block_count * (block_size / 2);
  for (int64_t step = 1; step < block_size; step <<= 1) {
    fht_step_blocks_kernel<<<launch_blocks_for(pair_count), threads, 0, stream>>>(
        work.data_ptr<float>(), block_size, pair_count, step);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

}  // namespace

std::vector<torch::Tensor> block_fht_forward_cuda(
    torch::Tensor latent,
    int64_t output_size,
    int64_t layers,
    int64_t seed,
    int64_t start,
    int64_t stop) {
  TORCH_CHECK(latent.is_contiguous(), "latent must be contiguous");
  TORCH_CHECK(start >= 0 && stop >= start && stop <= output_size, "invalid slice");
  TORCH_CHECK(layers >= 1 && layers <= 3, "layers must be 1, 2, or 3");
  int64_t latent_size = latent.numel();
  int64_t block_size = next_power_of_two(latent_size);
  TORCH_CHECK(block_size >= 32 && block_size <= kGlobalMaxBlockSize,
              "Block-FHT CUDA supports power-of-two block sizes from 2^5 to 2^23 after padding");
  auto out = torch::empty({stop - start}, latent.options());
  int64_t first_block = start / block_size;
  int64_t last_block = (stop + block_size - 1) / block_size;
  int64_t blocks = last_block - first_block;
  int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (block_size <= kSharedMaxBlockSize) {
    size_t smem = padded_shared_size(block_size) * sizeof(float);
    if (smem >= 48 * 1024) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          block_fht_forward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    }
    block_fht_forward_kernel<<<blocks, threads, smem, stream>>>(
        latent.data_ptr<float>(), out.data_ptr<float>(), (int)latent_size, (int)output_size,
        (int)block_size, (int)layers, (uint64_t)seed, (int)start, (int)stop);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    int64_t total = blocks * block_size;
    auto work = torch::empty({total}, latent.options());
    init_forward_blocks_kernel<<<launch_blocks_for(total), threads, 0, stream>>>(
        latent.data_ptr<float>(), work.data_ptr<float>(), latent_size, block_size,
        first_block, blocks, (uint64_t)seed);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    for (int layer = 0; layer < layers; ++layer) {
      run_global_fht_unscaled(work, block_size, blocks, stream);
      float scale = 1.0f / std::sqrt((float)block_size);
      apply_sign_scale_blocks_kernel<<<launch_blocks_for(total), threads, 0, stream>>>(
          work.data_ptr<float>(), block_size, first_block, blocks, layer + 1, (uint64_t)seed, scale);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    gather_slice_kernel<<<launch_blocks_for(stop - start), threads, 0, stream>>>(
        work.data_ptr<float>(), out.data_ptr<float>(), output_size, block_size,
        first_block, start, stop);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {out};
}

torch::Tensor block_fht_backward_cuda(
    torch::Tensor grad_out,
    int64_t latent_size,
    int64_t output_size,
    int64_t layers,
    int64_t seed,
    int64_t start,
    int64_t stop) {
  TORCH_CHECK(grad_out.is_contiguous(), "grad_out must be contiguous");
  int64_t block_size = next_power_of_two(latent_size);
  TORCH_CHECK(block_size >= 32 && block_size <= kGlobalMaxBlockSize,
              "Block-FHT CUDA supports power-of-two block sizes from 2^5 to 2^23 after padding");
  auto grad_latent = torch::zeros({latent_size}, grad_out.options());
  int64_t first_block = start / block_size;
  int64_t last_block = (stop + block_size - 1) / block_size;
  int64_t blocks = last_block - first_block;
  int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (block_size <= kSharedMaxBlockSize) {
    size_t smem = padded_shared_size(block_size) * sizeof(float);
    if (smem >= 48 * 1024) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          block_fht_backward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    }
    block_fht_backward_kernel<<<blocks, threads, smem, stream>>>(
        grad_out.data_ptr<float>(), grad_latent.data_ptr<float>(), (int)latent_size, (int)output_size,
        (int)block_size, (int)layers, (uint64_t)seed, (int)start, (int)stop);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    int64_t total = blocks * block_size;
    auto work = torch::empty({total}, grad_out.options());
    init_backward_blocks_kernel<<<launch_blocks_for(total), threads, 0, stream>>>(
        grad_out.data_ptr<float>(), work.data_ptr<float>(), output_size, block_size,
        first_block, blocks, start, stop);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    for (int layer = (int)layers - 1; layer >= 0; --layer) {
      if (layer == (int)layers - 1) {
        apply_sign_blocks_kernel<<<launch_blocks_for(total), threads, 0, stream>>>(
            work.data_ptr<float>(), block_size, first_block, blocks, layer + 1, (uint64_t)seed);
      } else {
        float scale = 1.0f / std::sqrt((float)block_size);
        apply_sign_scale_blocks_kernel<<<launch_blocks_for(total), threads, 0, stream>>>(
            work.data_ptr<float>(), block_size, first_block, blocks, layer + 1, (uint64_t)seed, scale);
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      run_global_fht_unscaled(work, block_size, blocks, stream);
    }
    float scale = 1.0f / std::sqrt((float)block_size);
    accumulate_latent_grad_scaled_kernel<<<launch_blocks_for(blocks * latent_size), threads, 0, stream>>>(
        work.data_ptr<float>(), grad_latent.data_ptr<float>(), latent_size, block_size,
        first_block, blocks, (uint64_t)seed, scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return grad_latent;
}

torch::Tensor block_fht_linear_forward_cuda(
    torch::Tensor input,
    torch::Tensor latent,
    int64_t out_features,
    int64_t layers,
    int64_t seed,
    double weight_scale) {
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(latent.is_contiguous(), "latent must be contiguous");
  TORCH_CHECK(input.dim() == 2, "input must be 2D [tokens, in_features]");
  TORCH_CHECK(layers >= 1 && layers <= 3, "layers must be 1, 2, or 3");
  int64_t tokens = input.size(0);
  int64_t in_features = input.size(1);
  int64_t latent_size = latent.numel();
  int64_t block_size = next_power_of_two(latent_size);
  TORCH_CHECK(block_size <= kSharedMaxBlockSize,
              "fused linear forward currently supports shared-memory block sizes <= 16384");
  TORCH_CHECK(block_size % in_features == 0,
              "fused linear forward requires block_size to be a multiple of in_features");
  TORCH_CHECK(out_features > 0, "out_features must be positive");
  TORCH_CHECK(in_features * out_features > 0, "linear weight must be non-empty");
  auto out = torch::zeros({tokens, out_features}, input.options());
  int64_t rows_per_block = block_size / in_features;
  int64_t blocks = (out_features + rows_per_block - 1) / rows_per_block;
  constexpr int kTokenTile = 8;
  dim3 grid((unsigned int)blocks, (unsigned int)((tokens + kTokenTile - 1) / kTokenTile));
  int threads = 256;
  size_t smem = padded_shared_size(block_size) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (smem >= 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        block_fht_linear_forward_kernel<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        block_fht_linear_forward_kernel<at::Half>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        block_fht_linear_forward_kernel<at::BFloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  }
  if (input.scalar_type() == torch::kFloat32) {
    block_fht_linear_forward_kernel<float><<<grid, threads, smem, stream>>>(
        input.data_ptr<float>(), latent.data_ptr<float>(), out.data_ptr<float>(),
        (int)tokens, (int)in_features, (int)out_features, (int)latent_size,
        (int)block_size, (int)layers, (uint64_t)seed, (float)weight_scale);
  } else if (input.scalar_type() == torch::kFloat16) {
    block_fht_linear_forward_kernel<at::Half><<<grid, threads, smem, stream>>>(
        input.data_ptr<at::Half>(), latent.data_ptr<at::Half>(), out.data_ptr<at::Half>(),
        (int)tokens, (int)in_features, (int)out_features, (int)latent_size,
        (int)block_size, (int)layers, (uint64_t)seed, (float)weight_scale);
  } else if (input.scalar_type() == torch::kBFloat16) {
    block_fht_linear_forward_kernel<at::BFloat16><<<grid, threads, smem, stream>>>(
        input.data_ptr<at::BFloat16>(), latent.data_ptr<at::BFloat16>(), out.data_ptr<at::BFloat16>(),
        (int)tokens, (int)in_features, (int)out_features, (int)latent_size,
        (int)block_size, (int)layers, (uint64_t)seed, (float)weight_scale);
  } else {
    TORCH_CHECK(false, "fused linear forward CUDA supports float32, float16, and bfloat16");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor fixed_basis_transform_cuda(
    torch::Tensor input,
    torch::Tensor permutations,
    torch::Tensor signs,
    int64_t block_size,
    bool inverse,
    bool shared_input) {
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(
      permutations.is_contiguous(), "permutations must be contiguous");
  TORCH_CHECK(signs.is_contiguous(), "signs must be contiguous");
  TORCH_CHECK(input.dim() == 2, "input must be 2D or flattened to 2D");
  TORCH_CHECK(
      permutations.dim() == 2,
      "permutations must have shape [bases, width]");
  TORCH_CHECK(signs.dim() == 2, "signs must have shape [bases, width]");
  TORCH_CHECK(
      permutations.sizes() == signs.sizes(),
      "permutations and signs must have identical shapes");
  TORCH_CHECK(
      block_size > 0 && (block_size & (block_size - 1)) == 0,
      "block_size must be a positive power of two");
  TORCH_CHECK(
      block_size <= kSharedMaxBlockSize,
      "fixed basis transform requires shared-memory block_size <= 16384");
  int64_t bases = permutations.size(0);
  int64_t width = permutations.size(1);
  TORCH_CHECK(width > 0, "width must be positive");
  TORCH_CHECK(
      width % block_size == 0, "block_size must divide width");
  int64_t tokens;
  if (shared_input) {
    TORCH_CHECK(
        input.size(1) == width,
        "shared input must have shape [tokens, width]");
    tokens = input.size(0);
  } else {
    TORCH_CHECK(
        input.size(0) % bases == 0 && input.size(1) == width,
        "per-basis input must flatten [bases, tokens, width]");
    tokens = input.size(0) / bases;
  }
  auto output = torch::empty({bases, tokens, width}, input.options());
  int64_t jobs = bases * tokens * (width / block_size);
  if (jobs == 0) {
    return output;
  }
  int device = -1;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties;
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  int64_t resident_blocks =
      (int64_t)properties.multiProcessorCount *
      (block_size == 256 ? 32 : 8);
  int grid = (int)std::min(jobs, resident_blocks);
  int threads = block_size == 256 ? 64 : 256;
  size_t smem = padded_shared_size(block_size) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (smem >= 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fixed_basis_transform_kernel<float>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fixed_basis_transform_kernel<at::Half>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fixed_basis_transform_kernel<at::BFloat16>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem));
  }
  if (input.scalar_type() == torch::kFloat32) {
    if (block_size == 256) {
      fixed_basis_transform_256_kernel<float>
          <<<grid, threads, smem, stream>>>(
              input.data_ptr<float>(),
              permutations.data_ptr<int64_t>(),
              signs.data_ptr<float>(),
              output.data_ptr<float>(),
              tokens,
              width,
              bases,
              jobs,
              inverse,
              shared_input);
    } else {
      fixed_basis_transform_kernel<float><<<grid, threads, smem, stream>>>(
        input.data_ptr<float>(),
        permutations.data_ptr<int64_t>(),
        signs.data_ptr<float>(),
        output.data_ptr<float>(),
        tokens,
        width,
        block_size,
        bases,
        jobs,
        inverse,
        shared_input);
    }
  } else if (input.scalar_type() == torch::kFloat16) {
    if (block_size == 256) {
      fixed_basis_transform_256_kernel<at::Half>
          <<<grid, threads, smem, stream>>>(
              input.data_ptr<at::Half>(),
              permutations.data_ptr<int64_t>(),
              signs.data_ptr<float>(),
              output.data_ptr<at::Half>(),
              tokens,
              width,
              bases,
              jobs,
              inverse,
              shared_input);
    } else {
      fixed_basis_transform_kernel<at::Half>
          <<<grid, threads, smem, stream>>>(
            input.data_ptr<at::Half>(),
            permutations.data_ptr<int64_t>(),
            signs.data_ptr<float>(),
            output.data_ptr<at::Half>(),
            tokens,
            width,
            block_size,
            bases,
            jobs,
            inverse,
            shared_input);
    }
  } else if (input.scalar_type() == torch::kBFloat16) {
    if (block_size == 256) {
      fixed_basis_transform_256_kernel<at::BFloat16>
          <<<grid, threads, smem, stream>>>(
              input.data_ptr<at::BFloat16>(),
              permutations.data_ptr<int64_t>(),
              signs.data_ptr<float>(),
              output.data_ptr<at::BFloat16>(),
              tokens,
              width,
              bases,
              jobs,
              inverse,
              shared_input);
    } else {
      fixed_basis_transform_kernel<at::BFloat16>
          <<<grid, threads, smem, stream>>>(
            input.data_ptr<at::BFloat16>(),
            permutations.data_ptr<int64_t>(),
            signs.data_ptr<float>(),
            output.data_ptr<at::BFloat16>(),
            tokens,
            width,
            block_size,
            bases,
            jobs,
            inverse,
            shared_input);
    }
  } else {
    TORCH_CHECK(
        false,
        "fixed basis transform supports float32, float16, and bfloat16");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor postgelu_mix_forward_cuda(
    torch::Tensor update,
    torch::Tensor condition,
    torch::Tensor slope,
    double scale) {
  TORCH_CHECK(
      update.is_contiguous() && condition.is_contiguous() &&
          slope.is_contiguous(),
      "postgelu_mix_forward inputs must be contiguous");
  TORCH_CHECK(
      update.dim() == 3 && condition.sizes() == update.sizes(),
      "postgelu_mix_forward expects matching [bases, tokens, width] tensors");
  TORCH_CHECK(
      slope.dim() == 2 && slope.size(0) == update.size(0) &&
          slope.size(1) == update.size(2),
      "postgelu_mix_forward slope must have shape [bases, width]");
  auto correction = torch::empty_like(update);
  int64_t elements = update.numel();
  int threads = 256;
  int blocks = (int)std::min<int64_t>(
      (elements + threads - 1) / threads, 65535);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      update.scalar_type(),
      "postgelu_mix_forward_cuda",
      [&] {
        postgelu_mix_forward_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                update.data_ptr<scalar_t>(),
                condition.data_ptr<scalar_t>(),
                slope.data_ptr<float>(),
                correction.data_ptr<scalar_t>(),
                elements,
                update.size(1),
                update.size(2),
                (float)scale);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return correction;
}

std::vector<torch::Tensor> postgelu_mix_backward_cuda(
    torch::Tensor grad_correction,
    torch::Tensor update,
    torch::Tensor condition,
    torch::Tensor slope,
    double scale) {
  TORCH_CHECK(
      grad_correction.is_contiguous() && update.is_contiguous() &&
          condition.is_contiguous() && slope.is_contiguous(),
      "postgelu_mix_backward inputs must be contiguous");
  TORCH_CHECK(
      grad_correction.sizes() == update.sizes() &&
          condition.sizes() == update.sizes(),
      "postgelu_mix_backward activation shapes must match");
  auto grad_update = torch::empty_like(update);
  auto grad_condition = torch::empty_like(condition);
  auto grad_slope = torch::zeros_like(slope);
  int64_t elements = update.numel();
  int threads = 256;
  int blocks = (int)std::min<int64_t>(
      (elements + threads - 1) / threads, 65535);
  constexpr int64_t token_chunks = 16;
  int64_t slope_blocks =
      update.size(0) * (update.size(2) + threads - 1) / threads *
      token_chunks;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      update.scalar_type(),
      "postgelu_mix_backward_cuda",
      [&] {
        postgelu_mix_backward_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                grad_correction.data_ptr<scalar_t>(),
                update.data_ptr<scalar_t>(),
                condition.data_ptr<scalar_t>(),
                slope.data_ptr<float>(),
                grad_update.data_ptr<scalar_t>(),
                grad_condition.data_ptr<scalar_t>(),
                elements,
                update.size(1),
                update.size(2),
                (float)scale);
        postgelu_slope_backward_kernel<scalar_t>
            <<<slope_blocks, threads, 0, stream>>>(
                grad_correction.data_ptr<scalar_t>(),
                update.data_ptr<scalar_t>(),
                condition.data_ptr<scalar_t>(),
                grad_slope.data_ptr<float>(),
                update.size(0),
                update.size(1),
                update.size(2),
                token_chunks,
                (float)scale);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_update, grad_condition, grad_slope};
}

torch::Tensor postgelu_sum_heads_cuda(
    torch::Tensor per_head,
    torch::Tensor residual,
    bool add_residual) {
  TORCH_CHECK(
      per_head.is_contiguous() && residual.is_contiguous(),
      "postgelu_sum_heads inputs must be contiguous");
  TORCH_CHECK(
      per_head.dim() == 3 && residual.dim() == 2 &&
          per_head.size(1) == residual.size(0) &&
          per_head.size(2) == residual.size(1),
      "postgelu_sum_heads expects [bases, tokens, width] and [tokens, width]");
  auto output = torch::empty_like(residual);
  int64_t elements = residual.numel();
  int threads = 256;
  int blocks = (int)std::min<int64_t>(
      (elements + threads - 1) / threads, 65535);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      per_head.scalar_type(),
      "postgelu_sum_heads_cuda",
      [&] {
        postgelu_sum_heads_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                per_head.data_ptr<scalar_t>(),
                residual.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                per_head.size(0),
                per_head.size(1),
                per_head.size(2),
                add_residual);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
