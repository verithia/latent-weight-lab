#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int64_t kValuesPerTile = 65536;

__global__ void pair_vq_partial_histogram_kernel(
    const float* __restrict__ values,
    const float* __restrict__ midpoints,
    int64_t values_per_row,
    int midpoint_count,
    int level_count,
    int tile_count,
    float* __restrict__ partial_sums,
    int32_t* __restrict__ partial_counts) {
  extern __shared__ unsigned char shared_bytes[];
  float* shared_sums = reinterpret_cast<float*>(shared_bytes);
  unsigned int* shared_counts = reinterpret_cast<unsigned int*>(
      shared_sums + level_count);
  for (int level = threadIdx.x; level < level_count; level += blockDim.x) {
    shared_sums[level] = 0.0f;
    shared_counts[level] = 0;
  }
  __syncthreads();

  const int row = blockIdx.y;
  const int tile = blockIdx.x;
  const float* row_values = values + static_cast<int64_t>(row) * values_per_row;
  const float* row_midpoints = midpoints + row * midpoint_count;
  const int64_t begin = static_cast<int64_t>(tile) * kValuesPerTile;
  const int64_t proposed_end = begin + kValuesPerTile;
  const int64_t end = proposed_end < values_per_row ? proposed_end : values_per_row;
  for (int64_t index = begin + threadIdx.x; index < end;
       index += blockDim.x) {
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
    atomicAdd(shared_sums + low, value);
    atomicAdd(shared_counts + low, 1U);
  }
  __syncthreads();

  const int64_t partial_base =
      (static_cast<int64_t>(row) * tile_count + tile) * level_count;
  for (int level = threadIdx.x; level < level_count; level += blockDim.x) {
    partial_sums[partial_base + level] = shared_sums[level];
    partial_counts[partial_base + level] =
        static_cast<int32_t>(shared_counts[level]);
  }
}

__global__ void pair_vq_reduce_partial_histogram_kernel(
    const float* __restrict__ partial_sums,
    const int32_t* __restrict__ partial_counts,
    int batch,
    int tile_count,
    int level_count,
    float* __restrict__ sums,
    int64_t* __restrict__ counts) {
  const int64_t total = static_cast<int64_t>(batch) * level_count;
  for (int64_t output = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
       output < total;
       output += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int row = static_cast<int>(output / level_count);
    const int level = static_cast<int>(output % level_count);
    float sum = 0.0f;
    int64_t count = 0;
    for (int tile = 0; tile < tile_count; ++tile) {
      const int64_t partial =
          (static_cast<int64_t>(row) * tile_count + tile) * level_count + level;
      sum += partial_sums[partial];
      count += partial_counts[partial];
    }
    sums[output] = sum;
    counts[output] = count;
  }
}

__global__ void pair_vq_partial_histogram_fp64_kernel(
    const float* __restrict__ values,
    const float* __restrict__ midpoints,
    int64_t values_per_row,
    int midpoint_count,
    int level_count,
    int tile_count,
    double* __restrict__ partial_sums,
    int32_t* __restrict__ partial_counts) {
  extern __shared__ unsigned char shared_bytes[];
  double* shared_sums = reinterpret_cast<double*>(shared_bytes);
  unsigned int* shared_counts = reinterpret_cast<unsigned int*>(
      shared_sums + level_count);
  for (int level = threadIdx.x; level < level_count; level += blockDim.x) {
    shared_sums[level] = 0.0;
    shared_counts[level] = 0;
  }
  __syncthreads();

  const int row = blockIdx.y;
  const int tile = blockIdx.x;
  const float* row_values = values + static_cast<int64_t>(row) * values_per_row;
  const float* row_midpoints = midpoints + row * midpoint_count;
  const int64_t begin = static_cast<int64_t>(tile) * kValuesPerTile;
  const int64_t proposed_end = begin + kValuesPerTile;
  const int64_t end = proposed_end < values_per_row ? proposed_end : values_per_row;
  for (int64_t index = begin + threadIdx.x; index < end;
       index += blockDim.x) {
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
    atomicAdd(shared_sums + low, static_cast<double>(value));
    atomicAdd(shared_counts + low, 1U);
  }
  __syncthreads();

  const int64_t partial_base =
      (static_cast<int64_t>(row) * tile_count + tile) * level_count;
  for (int level = threadIdx.x; level < level_count; level += blockDim.x) {
    partial_sums[partial_base + level] = shared_sums[level];
    partial_counts[partial_base + level] =
        static_cast<int32_t>(shared_counts[level]);
  }
}

__global__ void pair_vq_reduce_partial_histogram_fp64_kernel(
    const double* __restrict__ partial_sums,
    const int32_t* __restrict__ partial_counts,
    int batch,
    int tile_count,
    int level_count,
    float* __restrict__ sums,
    int64_t* __restrict__ counts) {
  const int64_t total = static_cast<int64_t>(batch) * level_count;
  for (int64_t output = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
       output < total;
       output += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int row = static_cast<int>(output / level_count);
    const int level = static_cast<int>(output % level_count);
    double sum = 0.0;
    int64_t count = 0;
    for (int tile = 0; tile < tile_count; ++tile) {
      const int64_t partial =
          (static_cast<int64_t>(row) * tile_count + tile) * level_count + level;
      sum += partial_sums[partial];
      count += partial_counts[partial];
    }
    sums[output] = static_cast<float>(sum);
    counts[output] = count;
  }
}

}  // namespace

std::vector<torch::Tensor> pair_vq_hierarchical_lloyd_stats_cuda(
    torch::Tensor values,
    torch::Tensor midpoints,
    int64_t level_count) {
  const int64_t batch = values.size(0);
  const int64_t values_per_row = values.size(1);
  const int tile_count = static_cast<int>(
      (values_per_row + kValuesPerTile - 1) / kValuesPerTile);
  auto partial_sums = torch::empty(
      {batch, tile_count, level_count},
      values.options().dtype(torch::kFloat32));
  auto partial_counts = torch::empty(
      {batch, tile_count, level_count},
      values.options().dtype(torch::kInt32));
  auto sums = torch::empty(
      {batch, level_count}, values.options().dtype(torch::kFloat32));
  auto counts = torch::empty(
      {batch, level_count}, values.options().dtype(torch::kInt64));

  const dim3 partial_grid(
      static_cast<unsigned int>(tile_count),
      static_cast<unsigned int>(batch));
  const size_t shared_bytes =
      static_cast<size_t>(level_count) * (sizeof(float) + sizeof(uint32_t));
  pair_vq_partial_histogram_kernel<<<
      partial_grid,
      kThreads,
      shared_bytes,
      at::cuda::getCurrentCUDAStream()>>>(
      values.data_ptr<float>(),
      midpoints.data_ptr<float>(),
      values_per_row,
      static_cast<int>(midpoints.size(1)),
      static_cast<int>(level_count),
      tile_count,
      partial_sums.data_ptr<float>(),
      partial_counts.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t output_count = batch * level_count;
  const int reduce_blocks = static_cast<int>(
      std::min<int64_t>((output_count + kThreads - 1) / kThreads, 4096));
  pair_vq_reduce_partial_histogram_kernel<<<
      std::max(reduce_blocks, 1),
      kThreads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      partial_sums.data_ptr<float>(),
      partial_counts.data_ptr<int32_t>(),
      static_cast<int>(batch),
      tile_count,
      static_cast<int>(level_count),
      sums.data_ptr<float>(),
      counts.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {sums, counts};
}

std::vector<torch::Tensor> pair_vq_hierarchical_lloyd_stats_fp64_cuda(
    torch::Tensor values,
    torch::Tensor midpoints,
    int64_t level_count) {
  const int64_t batch = values.size(0);
  const int64_t values_per_row = values.size(1);
  const int tile_count = static_cast<int>(
      (values_per_row + kValuesPerTile - 1) / kValuesPerTile);
  auto partial_sums = torch::empty(
      {batch, tile_count, level_count},
      values.options().dtype(torch::kFloat64));
  auto partial_counts = torch::empty(
      {batch, tile_count, level_count},
      values.options().dtype(torch::kInt32));
  auto sums = torch::empty(
      {batch, level_count}, values.options().dtype(torch::kFloat32));
  auto counts = torch::empty(
      {batch, level_count}, values.options().dtype(torch::kInt64));

  const dim3 partial_grid(
      static_cast<unsigned int>(tile_count),
      static_cast<unsigned int>(batch));
  const size_t shared_bytes =
      static_cast<size_t>(level_count) * (sizeof(double) + sizeof(uint32_t));
  pair_vq_partial_histogram_fp64_kernel<<<
      partial_grid,
      kThreads,
      shared_bytes,
      at::cuda::getCurrentCUDAStream()>>>(
      values.data_ptr<float>(),
      midpoints.data_ptr<float>(),
      values_per_row,
      static_cast<int>(midpoints.size(1)),
      static_cast<int>(level_count),
      tile_count,
      partial_sums.data_ptr<double>(),
      partial_counts.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t output_count = batch * level_count;
  const int reduce_blocks = static_cast<int>(
      std::min<int64_t>((output_count + kThreads - 1) / kThreads, 4096));
  pair_vq_reduce_partial_histogram_fp64_kernel<<<
      std::max(reduce_blocks, 1),
      kThreads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      partial_sums.data_ptr<double>(),
      partial_counts.data_ptr<int32_t>(),
      static_cast<int>(batch),
      tile_count,
      static_cast<int>(level_count),
      sums.data_ptr<float>(),
      counts.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {sums, counts};
}
