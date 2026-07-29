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

torch::Tensor block_fht_linear_forward_cuda(
    torch::Tensor input,
    torch::Tensor latent,
    int64_t out_features,
    int64_t layers,
    int64_t seed,
    double weight_scale);

torch::Tensor fixed_basis_transform_cuda(
    torch::Tensor input,
    torch::Tensor permutations,
    torch::Tensor signs,
    int64_t block_size,
    bool inverse,
    bool shared_input);

torch::Tensor postgelu_mix_forward_cuda(
    torch::Tensor update,
    torch::Tensor condition,
    torch::Tensor slope,
    double scale);

std::vector<torch::Tensor> postgelu_mix_backward_cuda(
    torch::Tensor grad_correction,
    torch::Tensor update,
    torch::Tensor condition,
    torch::Tensor slope,
    double scale);

torch::Tensor postgelu_sum_heads_cuda(
    torch::Tensor per_head,
    torch::Tensor residual,
    bool add_residual);

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

torch::Tensor block_fht_linear_forward(
    torch::Tensor input,
    torch::Tensor latent,
    int64_t out_features,
    int64_t layers,
    int64_t seed,
    double weight_scale) {
  TORCH_CHECK(input.is_cuda(), "block_fht_linear_forward: input must be CUDA");
  TORCH_CHECK(latent.is_cuda(), "block_fht_linear_forward: latent must be CUDA");
  TORCH_CHECK(input.scalar_type() == latent.scalar_type(), "block_fht_linear_forward: input and latent dtype must match");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat16 ||
                  input.scalar_type() == torch::kBFloat16,
              "block_fht_linear_forward: only float32, float16, and bfloat16 currently supported");
  return block_fht_linear_forward_cuda(input, latent, out_features, layers, seed, weight_scale);
}

torch::Tensor fixed_basis_transform(
    torch::Tensor input,
    torch::Tensor permutations,
    torch::Tensor signs,
    int64_t block_size,
    bool inverse,
    bool shared_input) {
  TORCH_CHECK(input.is_cuda(), "fixed_basis_transform: input must be CUDA");
  TORCH_CHECK(permutations.is_cuda(),
              "fixed_basis_transform: permutations must be CUDA");
  TORCH_CHECK(signs.is_cuda(), "fixed_basis_transform: signs must be CUDA");
  TORCH_CHECK(
      input.scalar_type() == torch::kFloat32 ||
          input.scalar_type() == torch::kFloat16 ||
          input.scalar_type() == torch::kBFloat16,
      "fixed_basis_transform: input must be float32, float16, or bfloat16");
  TORCH_CHECK(
      permutations.scalar_type() == torch::kInt64,
      "fixed_basis_transform: permutations must be int64");
  TORCH_CHECK(
      signs.scalar_type() == torch::kFloat32,
      "fixed_basis_transform: signs must be float32");
  return fixed_basis_transform_cuda(
      input,
      permutations,
      signs,
      block_size,
      inverse,
      shared_input);
}

torch::Tensor postgelu_mix_forward(
    torch::Tensor update,
    torch::Tensor condition,
    torch::Tensor slope,
    double scale) {
  TORCH_CHECK(update.is_cuda(), "postgelu_mix_forward: update must be CUDA");
  TORCH_CHECK(
      condition.is_cuda(), "postgelu_mix_forward: condition must be CUDA");
  TORCH_CHECK(slope.is_cuda(), "postgelu_mix_forward: slope must be CUDA");
  TORCH_CHECK(
      update.scalar_type() == condition.scalar_type(),
      "postgelu_mix_forward: update and condition dtypes must match");
  TORCH_CHECK(
      slope.scalar_type() == torch::kFloat32,
      "postgelu_mix_forward: slope must be float32");
  return postgelu_mix_forward_cuda(update, condition, slope, scale);
}

std::vector<torch::Tensor> postgelu_mix_backward(
    torch::Tensor grad_correction,
    torch::Tensor update,
    torch::Tensor condition,
    torch::Tensor slope,
    double scale) {
  TORCH_CHECK(
      grad_correction.is_cuda(),
      "postgelu_mix_backward: grad_correction must be CUDA");
  TORCH_CHECK(update.is_cuda(), "postgelu_mix_backward: update must be CUDA");
  TORCH_CHECK(
      condition.is_cuda(), "postgelu_mix_backward: condition must be CUDA");
  TORCH_CHECK(slope.is_cuda(), "postgelu_mix_backward: slope must be CUDA");
  TORCH_CHECK(
      grad_correction.scalar_type() == update.scalar_type() &&
          update.scalar_type() == condition.scalar_type(),
      "postgelu_mix_backward: activation dtypes must match");
  TORCH_CHECK(
      slope.scalar_type() == torch::kFloat32,
      "postgelu_mix_backward: slope must be float32");
  return postgelu_mix_backward_cuda(
      grad_correction, update, condition, slope, scale);
}

torch::Tensor postgelu_sum_heads(
    torch::Tensor per_head,
    torch::Tensor residual,
    bool add_residual) {
  TORCH_CHECK(
      per_head.is_cuda(), "postgelu_sum_heads: per_head must be CUDA");
  TORCH_CHECK(
      residual.is_cuda(), "postgelu_sum_heads: residual must be CUDA");
  TORCH_CHECK(
      per_head.scalar_type() == residual.scalar_type(),
      "postgelu_sum_heads: per_head and residual dtypes must match");
  return postgelu_sum_heads_cuda(per_head, residual, add_residual);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &block_fht_forward, "Block-FHT slice forward (CUDA)");
  m.def("backward", &block_fht_backward, "Block-FHT slice backward (CUDA)");
  m.def("linear_forward", &block_fht_linear_forward, "Block-FHT fused linear forward (CUDA)");
  m.def(
      "fixed_basis_transform",
      &fixed_basis_transform,
      "Batched fixed signed/permuted block-Hadamard transform (CUDA)");
  m.def(
      "postgelu_mix_forward",
      &postgelu_mix_forward,
      "Fused post-GELU multihead correction (CUDA)");
  m.def(
      "postgelu_mix_backward",
      &postgelu_mix_backward,
      "Fused post-GELU multihead correction backward (CUDA)");
  m.def(
      "postgelu_sum_heads",
      &postgelu_sum_heads,
      "Fused post-GELU head reduction with optional residual (CUDA)");
}
