import torch

from examples.nanogpt.analyze_mlp_lowbit_complete_neuron_functional import (
    FunctionalBank,
    SharedCompleteNeuronBank,
    complete_neuron_jvp,
    complete_neuron_output,
    decode_symmetric_int4,
    deployment_accounting,
    encode_symmetric_int4,
    fit_student,
    pack_signed_int4,
    relative_rmse,
    retained_centered_energy,
    terminal_shared_artifact,
    unpack_signed_int4,
)


def test_int4_pack_roundtrip_and_dequantization() -> None:
    value = torch.tensor(
        [[-1.0, -0.25, 0.0, 0.6], [0.1, 0.2, -0.3, 0.4]],
        dtype=torch.float32,
    )
    codes, scales = encode_symmetric_int4(value)
    packed = pack_signed_int4(codes)
    restored_codes = unpack_signed_int4(packed, value.numel()).reshape_as(codes)
    assert torch.equal(restored_codes, codes)
    restored = decode_symmetric_int4(
        packed, scales, tuple(value.shape), device="cpu"
    )
    assert torch.allclose(restored, codes.float() * scales.float())


def test_complete_neuron_jvp_matches_finite_difference() -> None:
    generator = torch.Generator().manual_seed(17)
    inputs = torch.randn(5, 4, generator=generator)
    directions = torch.randn(5, 4, generator=generator)
    detector = torch.randn(7, 4, generator=generator)
    write = torch.randn(7, 4, generator=generator)
    gain = torch.randn(7, generator=generator)
    analytic = complete_neuron_jvp(inputs, directions, detector, write, gain)
    epsilon = 1e-3
    positive = complete_neuron_output(
        inputs + epsilon * directions, detector, write, gain
    )
    negative = complete_neuron_output(
        inputs - epsilon * directions, detector, write, gain
    )
    numeric = (positive - negative) / (2 * epsilon)
    assert torch.allclose(analytic, numeric, atol=2e-3, rtol=2e-3)


def test_unquantized_complete_bank_preserves_pairing_and_identity() -> None:
    generator = torch.Generator().manual_seed(23)
    detector = torch.randn(6, 4, generator=generator)
    write = torch.randn(6, 4, generator=generator)
    sources = torch.zeros(6, dtype=torch.long)
    bank = SharedCompleteNeuronBank(
        detector, write, sources, [0], train_atoms=True
    )
    inputs = torch.randn(8, 4, generator=generator)
    expected = complete_neuron_output(
        inputs, detector, write, torch.ones(6)
    )
    actual, _ = bank.forward_function(0, inputs, None, quantized=False)
    assert relative_rmse(expected, actual) == 0.0
    assert retained_centered_energy(expected, actual) == 1.0


def test_exact_h47_payload_accounting() -> None:
    accounting = deployment_accounting(1421, 12, 768)
    assert accounting["int4_atom_values"] == 2_182_656
    assert accounting["int4_atom_bytes"] == 1_091_328
    assert accounting["fp16_scale_bytes"] == 5_684
    assert accounting["fp16_gain_bytes"] == 34_104
    assert accounting["total_checkpoint_payload_bytes"] == 1_131_116
    assert accounting["dense_replaced_mlp_fp16_bytes"] == 113_246_208


def test_terminal_artifact_exact_payload_and_shapes() -> None:
    detector = torch.linspace(-1, 1, 24).reshape(6, 4)
    write = torch.linspace(1, -1, 24).reshape(6, 4)
    sources = torch.tensor([0, 0, 0, 1, 1, 1])
    bank = SharedCompleteNeuronBank(
        detector, write, sources, [0, 1], train_atoms=True
    )
    accounting = deployment_accounting(6, 2, 4)
    artifact = terminal_shared_artifact(bank, accounting)
    assert artifact["accounted_payload_bytes"] == accounting[
        "total_checkpoint_payload_bytes"
    ]
    assert artifact["u_shape"] == [6, 4]
    assert artifact["v_shape"] == [6, 4]


def test_tiny_fit_updates_paired_student_without_nan() -> None:
    generator = torch.Generator().manual_seed(29)
    detector = torch.randn(6, 4, generator=generator)
    write = torch.randn(6, 4, generator=generator)
    inputs = torch.randn(16, 4, generator=generator)
    outputs = complete_neuron_output(
        inputs, detector, write, torch.ones(6)
    ).detach()
    source_layers = torch.zeros(6, dtype=torch.long)
    student = SharedCompleteNeuronBank(
        detector * 0.9,
        write * 1.1,
        source_layers,
        [0],
        train_atoms=True,
    )
    history = fit_student(
        student,
        FunctionalBank(inputs={0: inputs}, outputs={0: outputs}),
        {0: detector},
        {0: write},
        layers=[0],
        iterations=2,
        minibatch_rows=8,
        jvp_seed=31,
        learning_rate_atoms=1e-3,
        learning_rate_gains=3e-3,
        weight_decay=1e-4,
        gradient_clip_norm=1.0,
        device="cpu",
    )
    assert history
    assert all(torch.isfinite(parameter).all() for parameter in student.parameters())
