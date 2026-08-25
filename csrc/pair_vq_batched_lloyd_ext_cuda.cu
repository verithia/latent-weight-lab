#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

__global__ void pair_vq_batched_lloyd_stats_kernel(
    const float* __restrict__ values,
    const float* __restrict__ midpoints,
    int64_t values_per_row,
    int midpoint_count,
    int level_count,
    float* __restrict__ sums,
    int64_t* __restrict__ counts) {
  const int row = blockIdx.y;
  const float* row_values = values + static_cast<int64_t>(row) * values_per_row;
  const float* row_midpoints = midpoints + row * midpoint_count;
  float* row_sums = sums + row * level_count;
  int64_t* row_counts = counts + row * level_count;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
       index < values_per_row;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float value = row_values[index];
    int low = 0;
    int high = midpoint_count;
    while (low < high) {
      const int middle = low + ((high - low) >> 1);
      if (row_midpoints[middle] < value) {
        low = middle + 1;
      } else {
        high = middle;
      }
    }
    atomicAdd(row_sums + low, value);
    atomicAdd(
        reinterpret_cast<unsigned long long*>(row_counts + low),
        static_cast<unsigned long long>(1));
  }
}

}  // namespace

std::vector<torch::Tensor> pair_vq_batched_lloyd_stats_cuda(
    torch::Tensor values,
    torch::Tensor midpoints,
    int64_t level_count) {
  const int64_t batch = values.size(0);
  auto sums = torch::zeros(
      {batch, level_count}, values.options().dtype(torch::kFloat32));
  auto counts = torch::zeros(
      {batch, level_count}, values.options().dtype(torch::kInt64));
  constexpr int threads = 256;
  const int64_t required_blocks =
      (values.size(1) + threads - 1) / threads;
  const int blocks = static_cast<int>(
      std::min<int64_t>(required_blocks, 4096));
  const dim3 grid(std::max(blocks, 1), static_cast<unsigned int>(batch));
  pair_vq_batched_lloyd_stats_kernel<<<
      grid,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      values.data_ptr<float>(),
      midpoints.data_ptr<float>(),
      values.size(1),
      static_cast<int>(midpoints.size(1)),
      static_cast<int>(level_count),
      sums.data_ptr<float>(),
      counts.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {sums, counts};
}
