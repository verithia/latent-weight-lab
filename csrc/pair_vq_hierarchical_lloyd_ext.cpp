#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> pair_vq_hierarchical_lloyd_stats_cuda(
    torch::Tensor values,
    torch::Tensor midpoints,
    int64_t level_count);

std::vector<torch::Tensor> pair_vq_hierarchical_lloyd_stats(
    torch::Tensor values,
    torch::Tensor midpoints,
    int64_t level_count) {
  TORCH_CHECK(values.is_cuda() && midpoints.is_cuda(),
              "hierarchical Lloyd inputs must be CUDA");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32 &&
                  midpoints.scalar_type() == torch::kFloat32,
              "hierarchical Lloyd inputs must be float32");
  TORCH_CHECK(values.dim() == 2 && midpoints.dim() == 2,
              "hierarchical Lloyd inputs must be matrices");
  TORCH_CHECK(values.is_contiguous() && midpoints.is_contiguous(),
              "hierarchical Lloyd inputs must be contiguous");
  TORCH_CHECK(values.size(0) == midpoints.size(0),
              "hierarchical Lloyd row counts must match");
  TORCH_CHECK(level_count >= 2 && level_count <= 256,
              "hierarchical Lloyd level count must be in [2, 256]");
  TORCH_CHECK(midpoints.size(1) == level_count - 1,
              "hierarchical Lloyd midpoint count does not match levels");
  return pair_vq_hierarchical_lloyd_stats_cuda(
      values, midpoints, level_count);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("stats", &pair_vq_hierarchical_lloyd_stats,
        "Hierarchical Pair-VQ Lloyd statistics (CUDA)");
}
