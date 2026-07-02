#include <torch/extension.h>

std::vector<torch::Tensor> block_fht_forward_cuda(
    torch::Tensor latent,
    int64_t output_size,
    int64_t layers,
    int64_t seed,
    int64_t start,
    int64_t stop);

torch::Tensor block_fht_backward_cuda(
    torch::Tensor grad_out,
    int64_t latent_size,
    int64_t output_size,
    int64_t layers,
    int64_t seed,
    int64_t start,
    int64_t stop);

std::vector<torch::Tensor> block_fht_forward(
    torch::Tensor latent,
    int64_t output_size,
    int64_t layers,
    int64_t seed,
    int64_t start,
    int64_t stop) {
  TORCH_CHECK(latent.is_cuda(), "block_fht_forward: latent must be CUDA");
  TORCH_CHECK(latent.scalar_type() == torch::kFloat32, "block_fht_forward: only float32 currently supported");
  return block_fht_forward_cuda(latent, output_size, layers, seed, start, stop);
}

torch::Tensor block_fht_backward(
    torch::Tensor grad_out,
    int64_t latent_size,
    int64_t output_size,
    int64_t layers,
    int64_t seed,
    int64_t start,
    int64_t stop) {
  TORCH_CHECK(grad_out.is_cuda(), "block_fht_backward: grad_out must be CUDA");
  TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32, "block_fht_backward: only float32 currently supported");
  return block_fht_backward_cuda(grad_out, latent_size, output_size, layers, seed, start, stop);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &block_fht_forward, "Block-FHT slice forward (CUDA)");
  m.def("backward", &block_fht_backward, "Block-FHT slice backward (CUDA)");
}
