#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {

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
  TORCH_CHECK(block_size <= 16384, "prototype CUDA kernel supports block_size <= 16384");
  auto out = torch::empty({stop - start}, latent.options());
  int64_t first_block = start / block_size;
  int64_t last_block = (stop + block_size - 1) / block_size;
  int64_t blocks = last_block - first_block;
  int threads = 256;
  size_t smem = block_size * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
  block_fht_forward_kernel<<<blocks, threads, smem, stream>>>(
      latent.data_ptr<float>(), out.data_ptr<float>(), (int)latent_size, (int)output_size,
      (int)block_size, (int)layers, (uint64_t)seed, (int)start, (int)stop);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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
  TORCH_CHECK(block_size <= 16384, "prototype CUDA kernel supports block_size <= 16384");
  auto grad_latent = torch::zeros({latent_size}, grad_out.options());
  int64_t first_block = start / block_size;
  int64_t last_block = (stop + block_size - 1) / block_size;
  int64_t blocks = last_block - first_block;
  int threads = 256;
  size_t smem = block_size * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
  block_fht_backward_kernel<<<blocks, threads, smem, stream>>>(
      grad_out.data_ptr<float>(), grad_latent.data_ptr<float>(), (int)latent_size, (int)output_size,
      (int)block_size, (int)layers, (uint64_t)seed, (int)start, (int)stop);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_latent;
}
