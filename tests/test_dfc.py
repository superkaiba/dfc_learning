"""
Tests for Dedicated Feature Crosscoders (DFCs).

These tests validate that the DFC architecture correctly enforces partition
constraints through gradient masking, especially ensuring that auxiliary loss
doesn't leak gradients to forbidden parameters.
"""

import pytest
import torch as th
import tempfile
from pathlib import Path

from dictionary_learning import (
    FeaturePartition,
    DedicatedFeatureCrossCoder,
    DedicatedFeatureBatchTopKCrossCoder,
)
from dictionary_learning.trainers.crosscoder import BatchTopKCrossCoderTrainer


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def small_partition():
    """Create a small feature partition for testing."""
    return FeaturePartition(dict_size=1000, model_a_exclusive_size=50, model_b_exclusive_size=50)


@pytest.fixture
def small_dfc():
    """Create a small DFC for testing."""
    return DedicatedFeatureBatchTopKCrossCoder(
        activation_dim=64,
        dict_size=1000,
        num_layers=2,
        k=50,
        model_a_exclusive_pct=0.05,  # 50 features
        model_b_exclusive_pct=0.05,  # 50 features
    )


@pytest.fixture
def fake_activations():
    """Create fake paired activations for testing."""
    return th.randn(16, 2, 64)  # (batch, num_layers, activation_dim)


@pytest.fixture
def temp_model_path():
    """Create a temporary path for saving/loading models."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_dfc.pt"


# ============================================================================
# 1. Basic Functionality Tests
# ============================================================================


def test_feature_partition_creation():
    """Verify partition indices are computed correctly."""
    partition = FeaturePartition(dict_size=1000, model_a_exclusive_size=50, model_b_exclusive_size=50)

    # Check sizes
    assert partition.dict_size == 1000
    assert partition.model_a_exclusive_size == 50
    assert partition.model_b_exclusive_size == 50
    assert partition.shared_size == 900

    # Check indices
    assert len(partition.model_a_indices) == 50
    assert len(partition.model_b_indices) == 50
    assert len(partition.shared_indices) == 900

    # Check ranges
    assert partition.model_a_indices[0] == 0
    assert partition.model_a_indices[-1] == 49
    assert partition.model_b_indices[0] == 50
    assert partition.model_b_indices[-1] == 99
    assert partition.shared_indices[0] == 100
    assert partition.shared_indices[-1] == 999

    # Check allowed indices
    assert len(partition.model_a_allowed_indices) == 950  # A-exclusive + shared
    assert len(partition.model_b_allowed_indices) == 950  # B-exclusive + shared


def test_feature_partition_from_percentages():
    """Verify percentage-based initialization works correctly."""
    partition = FeaturePartition.from_percentages(1000, 0.05, 0.05)

    assert partition.model_a_exclusive_size == 50
    assert partition.model_b_exclusive_size == 50
    assert partition.shared_size == 900


def test_partition_masks(small_partition):
    """Verify encoder/decoder masks are correct."""
    # Model A (layer 0) mask
    mask_a = small_partition.get_encoder_mask(0)
    assert mask_a.shape == (1000,)
    assert mask_a[:50].all()  # A-exclusive features allowed
    assert not mask_a[50:100].any()  # B-exclusive features forbidden
    assert mask_a[100:].all()  # Shared features allowed

    # Model B (layer 1) mask
    mask_b = small_partition.get_encoder_mask(1)
    assert not mask_b[:50].any()  # A-exclusive features forbidden
    assert mask_b[50:100].all()  # B-exclusive features allowed
    assert mask_b[100:].all()  # Shared features allowed


def test_dfc_initialization(small_dfc):
    """Verify DFC initializes correctly."""
    assert small_dfc.activation_dim == 64
    assert small_dfc.dict_size == 1000
    assert small_dfc.num_layers == 2
    assert small_dfc.k == 50
    assert hasattr(small_dfc, 'partition')
    assert small_dfc.partition.model_a_exclusive_size == 50
    assert small_dfc.partition.model_b_exclusive_size == 50


def test_dfc_forward_pass(small_dfc, fake_activations):
    """Verify forward pass completes without errors."""
    with th.no_grad():
        x_hat, features = small_dfc(fake_activations, output_features=True)

    assert x_hat.shape == fake_activations.shape
    assert features.shape == (16, 1000)  # (batch, dict_size)


# ============================================================================
# 2. Weight Initialization Tests (Critical)
# ============================================================================


def test_forbidden_weights_initialized_to_zero(small_dfc):
    """CRITICAL: Verify forbidden encoder/decoder weights start at zero."""
    # Check encoder weights
    encoder_weight = small_dfc.encoder.weight  # (num_layers, activation_dim, dict_size)

    # Model A (layer 0) should have zero weights for B-exclusive features (50-99)
    model_a_b_exclusive = encoder_weight[0, :, 50:100]
    assert th.allclose(model_a_b_exclusive, th.zeros_like(model_a_b_exclusive), atol=1e-7), \
        "Model A encoder has non-zero weights for B-exclusive features"

    # Model B (layer 1) should have zero weights for A-exclusive features (0-49)
    model_b_a_exclusive = encoder_weight[1, :, :50]
    assert th.allclose(model_b_a_exclusive, th.zeros_like(model_b_a_exclusive), atol=1e-7), \
        "Model B encoder has non-zero weights for A-exclusive features"

    # Check decoder weights
    decoder_weight = small_dfc.decoder.weight  # (num_layers, dict_size, activation_dim)

    # Model A (layer 0) should have zero weights for B-exclusive features (50-99)
    model_a_b_exclusive_dec = decoder_weight[0, 50:100, :]
    assert th.allclose(model_a_b_exclusive_dec, th.zeros_like(model_a_b_exclusive_dec), atol=1e-7), \
        "Model A decoder has non-zero weights for B-exclusive features"

    # Model B (layer 1) should have zero weights for A-exclusive features (0-49)
    model_b_a_exclusive_dec = decoder_weight[1, :50, :]
    assert th.allclose(model_b_a_exclusive_dec, th.zeros_like(model_b_a_exclusive_dec), atol=1e-7), \
        "Model B decoder has non-zero weights for A-exclusive features"


def test_allowed_weights_nonzero(small_dfc):
    """Verify allowed weights are initialized with non-zero values."""
    encoder_weight = small_dfc.encoder.weight

    # Model A should have non-zero weights for A-exclusive features
    model_a_a_exclusive = encoder_weight[0, :, :50]
    assert not th.allclose(model_a_a_exclusive, th.zeros_like(model_a_a_exclusive)), \
        "Model A encoder has all-zero weights for A-exclusive features (should be initialized)"

    # Model A should have non-zero weights for shared features
    model_a_shared = encoder_weight[0, :, 100:]
    assert not th.allclose(model_a_shared, th.zeros_like(model_a_shared)), \
        "Model A encoder has all-zero weights for shared features (should be initialized)"


def test_partition_integrity_after_init(small_dfc):
    """Verify partition integrity check passes after initialization."""
    integrity = small_dfc.verify_partition_integrity()

    # All violations should be near zero
    for layer_key, violation in integrity['encoder'].items():
        assert violation < 1e-6, f"Encoder {layer_key} has violation {violation}"

    for layer_key, violation in integrity['decoder'].items():
        assert violation < 1e-6, f"Decoder {layer_key} has violation {violation}"


# ============================================================================
# 3. Gradient Masking Tests (Most Critical)
# ============================================================================


def test_gradient_masking_encoder(small_dfc, fake_activations):
    """CRITICAL: Verify encoder gradients are zeroed for forbidden features."""
    # Forward pass
    x_hat = small_dfc(fake_activations)

    # Compute loss and backward
    loss = (x_hat - fake_activations).pow(2).sum()
    loss.backward()

    # Check encoder gradients
    encoder_grad = small_dfc.encoder.weight.grad  # (num_layers, activation_dim, dict_size)

    # Model A (layer 0) should have zero gradients for B-exclusive features
    model_a_b_exclusive_grad = encoder_grad[0, :, 50:100]
    assert th.allclose(model_a_b_exclusive_grad, th.zeros_like(model_a_b_exclusive_grad), atol=1e-7), \
        f"Model A encoder has non-zero gradients for B-exclusive features: max={model_a_b_exclusive_grad.abs().max()}"

    # Model B (layer 1) should have zero gradients for A-exclusive features
    model_b_a_exclusive_grad = encoder_grad[1, :, :50]
    assert th.allclose(model_b_a_exclusive_grad, th.zeros_like(model_b_a_exclusive_grad), atol=1e-7), \
        f"Model B encoder has non-zero gradients for A-exclusive features: max={model_b_a_exclusive_grad.abs().max()}"


def test_gradient_masking_decoder(small_dfc, fake_activations):
    """CRITICAL: Verify decoder gradients are zeroed for forbidden features."""
    # Forward pass
    x_hat = small_dfc(fake_activations)

    # Compute loss and backward
    loss = (x_hat - fake_activations).pow(2).sum()
    loss.backward()

    # Check decoder gradients
    decoder_grad = small_dfc.decoder.weight.grad  # (num_layers, dict_size, activation_dim)

    # Model A (layer 0) should have zero gradients for B-exclusive features
    model_a_b_exclusive_grad = decoder_grad[0, 50:100, :]
    assert th.allclose(model_a_b_exclusive_grad, th.zeros_like(model_a_b_exclusive_grad), atol=1e-7), \
        f"Model A decoder has non-zero gradients for B-exclusive features: max={model_a_b_exclusive_grad.abs().max()}"

    # Model B (layer 1) should have zero gradients for A-exclusive features
    model_b_a_exclusive_grad = decoder_grad[1, :50, :]
    assert th.allclose(model_b_a_exclusive_grad, th.zeros_like(model_b_a_exclusive_grad), atol=1e-7), \
        f"Model B decoder has non-zero gradients for A-exclusive features: max={model_b_a_exclusive_grad.abs().max()}"


def test_gradients_flow_to_allowed_features(small_dfc, fake_activations):
    """Verify gradients DO reach allowed features (sanity check)."""
    # Forward pass
    x_hat = small_dfc(fake_activations)

    # Compute loss and backward
    loss = (x_hat - fake_activations).pow(2).sum()
    loss.backward()

    # Check that SOME gradients flow to allowed features
    encoder_grad = small_dfc.encoder.weight.grad

    # Model A should have non-zero gradients for A-exclusive features
    model_a_a_exclusive_grad = encoder_grad[0, :, :50]
    assert not th.allclose(model_a_a_exclusive_grad, th.zeros_like(model_a_a_exclusive_grad)), \
        "Model A encoder has all-zero gradients for A-exclusive features (gradients should flow)"

    # Model A should have non-zero gradients for shared features
    model_a_shared_grad = encoder_grad[0, :, 100:]
    assert not th.allclose(model_a_shared_grad, th.zeros_like(model_a_shared_grad)), \
        "Model A encoder has all-zero gradients for shared features (gradients should flow)"


def test_weights_stay_zero_after_backward(small_dfc, fake_activations):
    """Verify forbidden weights remain zero after backprop (before optimizer step)."""
    # Forward pass
    x_hat = small_dfc(fake_activations)

    # Compute loss and backward
    loss = (x_hat - fake_activations).pow(2).sum()
    loss.backward()

    # Weights should still be zero (gradients are masked, but weights unchanged)
    encoder_weight = small_dfc.encoder.weight

    # Model A should still have zero weights for B-exclusive features
    model_a_b_exclusive = encoder_weight[0, :, 50:100]
    assert th.allclose(model_a_b_exclusive, th.zeros_like(model_a_b_exclusive), atol=1e-7), \
        "Model A encoder weights changed for B-exclusive features after backward"


# ============================================================================
# 4. Auxiliary Loss Gradient Isolation Test (MOST CRITICAL)
# ============================================================================


def test_auxiliary_loss_no_gradient_leakage(small_dfc, fake_activations):
    """
    MOST CRITICAL TEST: Verify auxiliary loss doesn't leak gradients to forbidden parameters.

    This test simulates the auxiliary loss computation used for dead feature resurrection
    and verifies that gradients don't reach the wrong model's exclusive features.
    """
    # Ensure model is in training mode
    small_dfc.train()
    small_dfc.zero_grad()

    # Encode activations
    features = small_dfc.encode(fake_activations, normalize_activations=False)

    # Simulate auxiliary loss: pick some "dead" features and compute reconstruction loss
    # In real training, this would be done by the trainer's get_auxiliary_loss method
    # Here we simulate it to test gradient flow

    # Pretend features 10-19 (A-exclusive) are dead and need resurrection
    dead_features = th.zeros_like(features)
    dead_features[:, 10:20] = th.randn(16, 10)  # Activate some A-exclusive features

    # Decode these features to get reconstruction
    x_reconstruct = small_dfc.decode(dead_features, denormalize_activations=False)

    # Compute auxiliary loss (reconstruction error for both models)
    aux_loss = (x_reconstruct - fake_activations).pow(2).sum()

    # Backward pass
    aux_loss.backward()

    # Now check gradients
    decoder_grad = small_dfc.decoder.weight.grad

    # CRITICAL CHECK: Model B (layer 1) should have ZERO gradients for the A-exclusive
    # features we just "resurrected" (indices 10-19), because gradient masking should
    # prevent gradients from reaching Model B's decoder for A-exclusive features
    model_b_a_exclusive_grad = decoder_grad[1, :50, :]  # All A-exclusive features for Model B
    assert th.allclose(model_b_a_exclusive_grad, th.zeros_like(model_b_a_exclusive_grad), atol=1e-7), \
        f"CRITICAL FAILURE: Auxiliary loss leaked gradients to Model B decoder for A-exclusive features! " \
        f"Max gradient: {model_b_a_exclusive_grad.abs().max():.2e}"

    # Reverse check: Model A should have gradients for its own exclusive features
    model_a_a_exclusive_grad = decoder_grad[0, :50, :]
    # At least SOME gradients should be non-zero for the features we activated
    activated_features_grad = decoder_grad[0, 10:20, :]
    assert not th.allclose(activated_features_grad, th.zeros_like(activated_features_grad)), \
        "Model A decoder has no gradients for activated A-exclusive features (should have gradients)"

    print("✓ Auxiliary loss gradient isolation test PASSED!")


def test_auxiliary_loss_both_directions(small_dfc, fake_activations):
    """Test auxiliary loss doesn't leak in either direction (A→B and B→A)."""
    small_dfc.train()

    # Test A→B: Activate B-exclusive features, check A doesn't get gradients
    small_dfc.zero_grad()
    dead_features_b = th.zeros(16, 1000)
    dead_features_b[:, 60:70] = th.randn(16, 10)  # B-exclusive features

    x_reconstruct = small_dfc.decode(dead_features_b, denormalize_activations=False)
    loss_b = (x_reconstruct - fake_activations).pow(2).sum()
    loss_b.backward()

    # Model A should have zero gradients for B-exclusive features
    decoder_grad = small_dfc.decoder.weight.grad
    model_a_b_exclusive_grad = decoder_grad[0, 50:100, :]
    assert th.allclose(model_a_b_exclusive_grad, th.zeros_like(model_a_b_exclusive_grad), atol=1e-7), \
        f"Model A got gradients for B-exclusive features: {model_a_b_exclusive_grad.abs().max():.2e}"

    # Test B→A: Activate A-exclusive features, check B doesn't get gradients
    small_dfc.zero_grad()
    dead_features_a = th.zeros(16, 1000)
    dead_features_a[:, 10:20] = th.randn(16, 10)  # A-exclusive features

    x_reconstruct = small_dfc.decode(dead_features_a, denormalize_activations=False)
    loss_a = (x_reconstruct - fake_activations).pow(2).sum()
    loss_a.backward()

    # Model B should have zero gradients for A-exclusive features
    decoder_grad = small_dfc.decoder.weight.grad
    model_b_a_exclusive_grad = decoder_grad[1, :50, :]
    assert th.allclose(model_b_a_exclusive_grad, th.zeros_like(model_b_a_exclusive_grad), atol=1e-7), \
        f"Model B got gradients for A-exclusive features: {model_b_a_exclusive_grad.abs().max():.2e}"

    print("✓ Bidirectional auxiliary loss isolation test PASSED!")


# ============================================================================
# 5. Optimizer State Tests
# ============================================================================


def test_optimizer_step_maintains_zero_weights(small_dfc, fake_activations):
    """Verify forbidden weights remain zero after optimizer step."""
    # Create optimizer
    optimizer = th.optim.Adam(small_dfc.parameters(), lr=1e-3)

    # Training step
    x_hat = small_dfc(fake_activations)
    loss = (x_hat - fake_activations).pow(2).sum()
    loss.backward()
    optimizer.step()

    # Check weights are still zero
    integrity = small_dfc.verify_partition_integrity()
    for layer_key, violation in integrity['encoder'].items():
        assert violation < 1e-6, f"After optimizer step: Encoder {layer_key} has violation {violation}"
    for layer_key, violation in integrity['decoder'].items():
        assert violation < 1e-6, f"After optimizer step: Decoder {layer_key} has violation {violation}"


def test_multiple_optimizer_steps_maintain_zero(small_dfc, fake_activations):
    """Verify forbidden weights remain zero over multiple training steps."""
    optimizer = th.optim.Adam(small_dfc.parameters(), lr=1e-3)

    for step in range(10):
        optimizer.zero_grad()
        x_hat = small_dfc(fake_activations)
        loss = (x_hat - fake_activations).pow(2).sum()
        loss.backward()
        optimizer.step()

    # Check weights are still zero after 10 steps
    integrity = small_dfc.verify_partition_integrity()
    max_violation = max(
        max(integrity['encoder'].values()),
        max(integrity['decoder'].values())
    )
    assert max_violation < 1e-5, f"After 10 steps, max violation is {max_violation}"


# ============================================================================
# 6. Serialization Tests
# ============================================================================


def test_save_and_load_preserves_partition(small_dfc, temp_model_path):
    """Verify partition config is saved and loaded correctly."""
    # Save model
    th.save(small_dfc.state_dict(), temp_model_path)

    # Load model
    loaded_dfc = DedicatedFeatureBatchTopKCrossCoder(
        activation_dim=64,
        dict_size=1000,
        num_layers=2,
        k=50,
        model_a_exclusive_pct=0.05,
        model_b_exclusive_pct=0.05,
    )
    loaded_dfc.load_state_dict(th.load(temp_model_path, weights_only=True))

    # Check partition is preserved
    assert loaded_dfc.partition.model_a_exclusive_size == 50
    assert loaded_dfc.partition.model_b_exclusive_size == 50
    assert loaded_dfc.partition.shared_size == 900


def test_partition_integrity_after_load(small_dfc, temp_model_path, fake_activations):
    """Verify weights are still zero after loading."""
    # Train for a few steps
    optimizer = th.optim.Adam(small_dfc.parameters(), lr=1e-3)
    for _ in range(5):
        optimizer.zero_grad()
        x_hat = small_dfc(fake_activations)
        loss = (x_hat - fake_activations).pow(2).sum()
        loss.backward()
        optimizer.step()

    # Save
    th.save(small_dfc.state_dict(), temp_model_path)

    # Load
    loaded_dfc = DedicatedFeatureBatchTopKCrossCoder(
        activation_dim=64,
        dict_size=1000,
        num_layers=2,
        k=50,
        model_a_exclusive_pct=0.05,
        model_b_exclusive_pct=0.05,
    )
    loaded_dfc.load_state_dict(th.load(temp_model_path, weights_only=True))

    # Check integrity
    integrity = loaded_dfc.verify_partition_integrity()
    max_violation = max(
        max(integrity['encoder'].values()),
        max(integrity['decoder'].values())
    )
    assert max_violation < 1e-5, f"After load, max violation is {max_violation}"


# ============================================================================
# 7. Statistics and Logging Tests
# ============================================================================


def test_get_partition_statistics(small_dfc):
    """Verify partition statistics are computed correctly."""
    stats = small_dfc.get_partition_statistics()

    assert stats['total_features'] == 1000
    assert stats['model_a_exclusive'] == 50
    assert stats['model_b_exclusive'] == 50
    assert stats['shared'] == 900
    assert abs(stats['model_a_exclusive_pct'] - 0.05) < 1e-6
    assert abs(stats['model_b_exclusive_pct'] - 0.05) < 1e-6
    assert abs(stats['shared_pct'] - 0.90) < 1e-6


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
