#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

__global__ void pair_vq_lloyd_stats_kernel(
    const float* __restrict__ values,
    const float* __restrict__ midpoints,
    int64_t value_count,
    int midpoint_count,
    float* __restrict__ sums,
    int64_t* __restrict__ counts) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
       index < value_count;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float value = values[index];
    int low = 0;
    int high = midpoint_count;
    while (low < high) {
      const int middle = low + ((high - low) >> 1);
      if (midpoints[middle] < value) {
        low = middle + 1;
      } else {
        high = middle;
      }
    }
    atomicAdd(sums + low, value);
    atomicAdd(
        reinterpret_cast<unsigned long long*>(counts + low),
        static_cast<unsigned long long>(1));
  }
}

}  // namespace

std::vector<torch::Tensor> pair_vq_lloyd_stats_cuda(
    torch::Tensor values,
    torch::Tensor midpoints,
    int64_t level_count) {
  auto sums = torch::zeros(
      {level_count}, values.options().dtype(torch::kFloat32));
  auto counts = torch::zeros(
      {level_count}, values.options().dtype(torch::kInt64));
  constexpr int threads = 256;
  const int64_t required_blocks =
      (values.numel() + threads - 1) / threads;
  const int blocks = static_cast<int>(
      std::min<int64_t>(required_blocks, 4096));
  pair_vq_lloyd_stats_kernel<<<
      std::max(blocks, 1),
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      values.data_ptr<float>(),
      midpoints.data_ptr<float>(),
      values.numel(),
      static_cast<int>(midpoints.numel()),
      sums.data_ptr<float>(),
      counts.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {sums, counts};
}
