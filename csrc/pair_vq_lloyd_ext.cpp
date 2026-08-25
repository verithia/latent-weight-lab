#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> pair_vq_lloyd_stats_cuda(
    torch::Tensor values,
    torch::Tensor midpoints,
    int64_t level_count);

std::vector<torch::Tensor> pair_vq_lloyd_stats(
    torch::Tensor values,
    torch::Tensor midpoints,
    int64_t level_count) {
  TORCH_CHECK(values.is_cuda(), "Pair-VQ Lloyd values must be CUDA");
  TORCH_CHECK(midpoints.is_cuda(), "Pair-VQ Lloyd midpoints must be CUDA");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32,
              "Pair-VQ Lloyd values must be float32");
  TORCH_CHECK(midpoints.scalar_type() == torch::kFloat32,
              "Pair-VQ Lloyd midpoints must be float32");
  TORCH_CHECK(values.dim() == 1 && values.is_contiguous(),
              "Pair-VQ Lloyd values must be a contiguous vector");
  TORCH_CHECK(midpoints.dim() == 1 && midpoints.is_contiguous(),
              "Pair-VQ Lloyd midpoints must be a contiguous vector");
  TORCH_CHECK(level_count >= 2 && level_count <= 256,
              "Pair-VQ Lloyd level count must be in [2, 256]");
  TORCH_CHECK(midpoints.numel() == level_count - 1,
              "Pair-VQ Lloyd midpoint count does not match levels");
  return pair_vq_lloyd_stats_cuda(values, midpoints, level_count);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("stats", &pair_vq_lloyd_stats,
        "Fused Pair-VQ Lloyd assignment statistics (CUDA)");
}
