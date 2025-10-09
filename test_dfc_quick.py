"""
Quick verification script for DFC implementation.
Tests the most critical functionality without heavy dependencies.
"""

import torch as th
import sys

print("Starting DFC quick verification...")
print(f"PyTorch version: {th.__version__}")
print()

# Test 1: Import DFC classes
print("Test 1: Importing DFC classes...")
try:
    from dictionary_learning import FeaturePartition, DedicatedFeatureBatchTopKCrossCoder
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create partition
print("\nTest 2: Creating FeaturePartition...")
try:
    partition = FeaturePartition.from_percentages(1000, 0.05, 0.05)
    assert partition.model_a_exclusive_size == 50
    assert partition.model_b_exclusive_size == 50
    assert partition.shared_size == 900
    print(f"✓ Partition created: {partition}")
except Exception as e:
    print(f"✗ Partition creation failed: {e}")
    sys.exit(1)

# Test 3: Create small DFC
print("\nTest 3: Creating DFC...")
try:
    dfc = DedicatedFeatureBatchTopKCrossCoder(
        activation_dim=64,
        dict_size=1000,
        num_layers=2,
        k=50,
        model_a_exclusive_pct=0.05,
        model_b_exclusive_pct=0.05,
    )
    print(f"✓ DFC created successfully")
except Exception as e:
    print(f"✗ DFC creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check forbidden weights are zero on init
print("\nTest 4: Checking forbidden weights are zero...")
try:
    encoder_weight = dfc.encoder.weight  # (num_layers, activation_dim, dict_size)
    decoder_weight = dfc.decoder.weight  # (num_layers, dict_size, activation_dim)

    # Model A (layer 0) should have zero weights for B-exclusive features (50-99)
    model_a_b_exclusive_enc = encoder_weight[0, :, 50:100]
    max_enc_violation = model_a_b_exclusive_enc.abs().max().item()

    model_a_b_exclusive_dec = decoder_weight[0, 50:100, :]
    max_dec_violation = model_a_b_exclusive_dec.abs().max().item()

    if max_enc_violation < 1e-6 and max_dec_violation < 1e-6:
        print(f"✓ Forbidden weights are zero (enc: {max_enc_violation:.2e}, dec: {max_dec_violation:.2e})")
    else:
        print(f"✗ Forbidden weights not zero! (enc: {max_enc_violation:.2e}, dec: {max_dec_violation:.2e})")
        sys.exit(1)
except Exception as e:
    print(f"✗ Weight check failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Forward pass
print("\nTest 5: Testing forward pass...")
try:
    fake_activations = th.randn(4, 2, 64)  # (batch, num_layers, activation_dim)
    with th.no_grad():
        x_hat, features = dfc(fake_activations, output_features=True)

    assert x_hat.shape == fake_activations.shape
    assert features.shape == (4, 1000)
    print(f"✓ Forward pass successful: x_hat.shape={x_hat.shape}, features.shape={features.shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Gradient masking
print("\nTest 6: Testing gradient masking (CRITICAL)...")
try:
    dfc.zero_grad()
    x_hat = dfc(fake_activations)
    loss = (x_hat - fake_activations).pow(2).sum()
    loss.backward()

    # Check encoder gradients
    encoder_grad = dfc.encoder.weight.grad
    model_a_b_exclusive_grad = encoder_grad[0, :, 50:100]
    max_grad = model_a_b_exclusive_grad.abs().max().item()

    if max_grad < 1e-6:
        print(f"✓ Encoder gradients masked correctly (max: {max_grad:.2e})")
    else:
        print(f"✗ Encoder gradients NOT masked! (max: {max_grad:.2e})")
        sys.exit(1)

    # Check decoder gradients
    decoder_grad = dfc.decoder.weight.grad
    model_a_b_exclusive_grad_dec = decoder_grad[0, 50:100, :]
    max_grad_dec = model_a_b_exclusive_grad_dec.abs().max().item()

    if max_grad_dec < 1e-6:
        print(f"✓ Decoder gradients masked correctly (max: {max_grad_dec:.2e})")
    else:
        print(f"✗ Decoder gradients NOT masked! (max: {max_grad_dec:.2e})")
        sys.exit(1)

except Exception as e:
    print(f"✗ Gradient masking test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Auxiliary loss gradient isolation (MOST CRITICAL)
print("\nTest 7: Testing auxiliary loss gradient isolation (MOST CRITICAL)...")
try:
    dfc.zero_grad()

    # Simulate auxiliary loss: activate some A-exclusive features
    dead_features = th.zeros(4, 1000)
    dead_features[:, 10:20] = th.randn(4, 10)  # Activate A-exclusive features

    # Decode and compute loss
    x_reconstruct = dfc.decode(dead_features, denormalize_activations=False)
    aux_loss = (x_reconstruct - fake_activations).pow(2).sum()
    aux_loss.backward()

    # Check that Model B doesn't get gradients for A-exclusive features
    decoder_grad = dfc.decoder.weight.grad
    model_b_a_exclusive_grad = decoder_grad[1, :50, :]  # Model B, A-exclusive features
    max_leak = model_b_a_exclusive_grad.abs().max().item()

    if max_leak < 1e-6:
        print(f"✓ Auxiliary loss doesn't leak gradients (max: {max_leak:.2e})")
    else:
        print(f"✗ CRITICAL FAILURE: Auxiliary loss leaked gradients! (max: {max_leak:.2e})")
        sys.exit(1)

except Exception as e:
    print(f"✗ Auxiliary loss test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Weights stay zero after optimizer step
print("\nTest 8: Testing weights stay zero after optimizer step...")
try:
    optimizer = th.optim.Adam(dfc.parameters(), lr=1e-3)

    dfc.zero_grad()
    x_hat = dfc(fake_activations)
    loss = (x_hat - fake_activations).pow(2).sum()
    loss.backward()
    optimizer.step()

    # Check weights are still zero
    integrity = dfc.verify_partition_integrity()
    max_violation = max(
        max(integrity['encoder'].values()),
        max(integrity['decoder'].values())
    )

    if max_violation < 1e-5:
        print(f"✓ Weights stay zero after optimizer step (max violation: {max_violation:.2e})")
    else:
        print(f"✗ Weights drifted from zero! (max violation: {max_violation:.2e})")
        sys.exit(1)

except Exception as e:
    print(f"✗ Optimizer step test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("ALL CRITICAL TESTS PASSED! ✓")
print("="*60)
print("\nThe DFC implementation correctly:")
print("  1. Initializes forbidden weights to zero")
print("  2. Masks gradients for forbidden features")
print("  3. Prevents auxiliary loss from leaking gradients")
print("  4. Maintains zero weights after optimizer steps")
print("\nThe implementation is ready for use!")
