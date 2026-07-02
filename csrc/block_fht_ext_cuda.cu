#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

namespace {

constexpr int64_t kSharedMaxBlockSize = 16384;
constexpr int64_t kGlobalMaxBlockSize = 1LL << 23;

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

__device__ void fht_inplace(float* values, int n) {
  for (int step = 1; step < n; step <<= 1) {
    __syncthreads();
    for (int i = threadIdx.x; i < n / 2; i += blockDim.x) {
      int group = i / step;
      int lane = i - group * step;
      int a = group * (step << 1) + lane;
      int b = a + step;
      float x = values[a];
      float y = values[b];
      values[a] = x + y;
      values[b] = x - y;
    }
  }
  __syncthreads();
  float scale = rsqrtf((float)n);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    values[i] *= scale;
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

  for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
    float v = i < latent_size ? latent[i] : 0.0f;
    values[i] = v * sign_for(seed, block, 0, i);
  }
  __syncthreads();

  for (int layer = 0; layer < layers; ++layer) {
    fht_inplace(values, block_size);
    for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
      values[i] *= sign_for(seed, block, layer + 1, i);
    }
    __syncthreads();
  }

  for (int i = local_start + threadIdx.x; i < local_stop; i += blockDim.x) {
    int global_index = block_start + i;
    if (global_index < output_size) {
      out[global_index - start] = values[i];
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
    values[i] = 0.0f;
  }
  __syncthreads();
  for (int i = local_start + threadIdx.x; i < local_stop; i += blockDim.x) {
    int global_index = block_start + i;
    if (global_index < output_size) {
      values[i] = grad_out[global_index - start];
    }
  }
  __syncthreads();

  for (int layer = layers - 1; layer >= 0; --layer) {
    for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
      values[i] *= sign_for(seed, block, layer + 1, i);
    }
    __syncthreads();
    fht_inplace(values, block_size);
  }
  for (int i = threadIdx.x; i < latent_size; i += blockDim.x) {
    float g = values[i] * sign_for(seed, block, 0, i);
    atomicAdd(grad_latent + i, g);
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

int launch_blocks_for(int64_t work_items) {
  int64_t blocks = (work_items + 255) / 256;
  if (blocks < 1) return 1;
  if (blocks > 65535) return 65535;
  return (int)blocks;
}

void run_global_fht(torch::Tensor work, int64_t block_size, int64_t block_count, cudaStream_t stream) {
  int threads = 256;
  int64_t pair_count = block_count * (block_size / 2);
  for (int64_t step = 1; step < block_size; step <<= 1) {
    fht_step_blocks_kernel<<<launch_blocks_for(pair_count), threads, 0, stream>>>(
        work.data_ptr<float>(), block_size, pair_count, step);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  int64_t total = block_count * block_size;
  float scale = 1.0f / std::sqrt((float)block_size);
  scale_blocks_kernel<<<launch_blocks_for(total), threads, 0, stream>>>(
      work.data_ptr<float>(), total, scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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
    size_t smem = block_size * sizeof(float);
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
      run_global_fht(work, block_size, blocks, stream);
      apply_sign_blocks_kernel<<<launch_blocks_for(total), threads, 0, stream>>>(
          work.data_ptr<float>(), block_size, first_block, blocks, layer + 1, (uint64_t)seed);
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
    size_t smem = block_size * sizeof(float);
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
      apply_sign_blocks_kernel<<<launch_blocks_for(total), threads, 0, stream>>>(
          work.data_ptr<float>(), block_size, first_block, blocks, layer + 1, (uint64_t)seed);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      run_global_fht(work, block_size, blocks, stream);
    }
    accumulate_latent_grad_kernel<<<launch_blocks_for(blocks * latent_size), threads, 0, stream>>>(
        work.data_ptr<float>(), grad_latent.data_ptr<float>(), latent_size, block_size,
        first_block, blocks, (uint64_t)seed);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return grad_latent;
}
