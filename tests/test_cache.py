import os
import tempfile
import torch as th
import pytest
from dictionary_learning.cache import ActivationShard, save_shard
from datasets import Dataset
from nnsight import LanguageModel
from dictionary_learning.cache import ActivationCache
from transformers import AutoModelForCausalLM, AutoTokenizer


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


def test_activation_shard_float16(temp_dir):
    # Create random activations
    dtype = th.float16
    shape = (100, 128)  # 100 tokens, 128 dimensions
    activations = th.randn(shape, dtype=dtype)

    # Save the activations
    save_shard(activations, temp_dir, 0, "test", "out")

    # Load the activations
    shard = ActivationShard(temp_dir, 0)

    # Check the shape and dtype
    assert shard.shape == shape
    assert shard.dtype == dtype

    # Check the content
    loaded_activations = shard[:]
    assert th.equal(activations, loaded_activations)


def test_activation_shard_bfloat16(temp_dir):
    # Create random activations
    dtype = th.bfloat16
    shape = (100, 128)  # 100 tokens, 128 dimensions
    activations = th.randn(shape, dtype=dtype)

    # Save the activations
    save_shard(activations, temp_dir, 0, "test", "out")

    # Load the activations
    shard = ActivationShard(temp_dir, 0)

    # Check the shape and dtype
    assert shard.shape == shape
    assert shard.dtype == dtype

    # Check the content
    loaded_activations = shard[:]
    assert th.equal(activations, loaded_activations)


def test_activation_shard_float32(temp_dir):
    # Create random activations
    dtype = th.float32
    shape = (100, 128)  # 100 tokens, 128 dimensions
    activations = th.randn(shape, dtype=dtype)

    # Save the activations
    save_shard(activations, temp_dir, 0, "test", "out")

    # Load the activations
    shard = ActivationShard(temp_dir, 0)

    # Check the shape and dtype
    assert shard.shape == shape
    assert shard.dtype == dtype

    # Check the content
    loaded_activations = shard[:]
    assert th.equal(activations, loaded_activations)


def test_activation_shard_int8(temp_dir):
    # Create random activations
    dtype = th.int8
    shape = (100, 128)  # 100 tokens, 128 dimensions
    activations = th.randint(-128, 127, shape, dtype=dtype)

    # Save the activations
    save_shard(activations, temp_dir, 0, "test", "out")

    # Load the activations
    shard = ActivationShard(temp_dir, 0)

    # Check the shape and dtype
    assert shard.shape == shape
    assert shard.dtype == dtype

    # Check the content
    loaded_activations = shard[:]
    assert th.all(activations == loaded_activations)


def test_activation_shard_indexing(temp_dir):
    # Create random activations
    dtype = th.float32
    shape = (100, 128)  # 100 tokens, 128 dimensions
    activations = th.randn(shape, dtype=dtype)

    # Save the activations
    save_shard(activations, temp_dir, 0, "test", "out")

    # Load the activations
    shard = ActivationShard(temp_dir, 0)

    # Test different indexing patterns
    # Single index
    assert th.equal(activations[5], shard[5])

    # Slice
    assert th.equal(activations[10:20], shard[10:20])

    # List of indices
    indices = [5, 10, 15, 20]
    assert th.equal(activations[indices], shard[indices])


def test_activation_shard_multiple_shards(temp_dir):
    # Create and save multiple shards
    dtype = th.float32
    shape = (100, 128)  # 100 tokens, 128 dimensions

    # Create and save shard 0
    activations0 = th.randn(shape, dtype=dtype)
    save_shard(activations0, temp_dir, 0, "test", "out")

    # Create and save shard 1
    activations1 = th.randn(shape, dtype=dtype)
    save_shard(activations1, temp_dir, 1, "test", "out")

    # Load shards
    shard0 = ActivationShard(temp_dir, 0)
    shard1 = ActivationShard(temp_dir, 1)

    # Verify contents
    assert th.equal(activations0, shard0[:])
    assert th.equal(activations1, shard1[:])


def test_activation_cache_with_normalizer(temp_dir):
    """Test ActivationCache collection and normalizer against direct model activations."""
    # Set flag to handle meta tensors properly
    th.fx.experimental._config.meta_nonzero_assume_all_nonzero = True

    # Skip test if CUDA not available to avoid device mapping issues
    if not th.cuda.is_available():
        pytest.skip("CUDA not available, skipping test to avoid device mapping issues")

    # Test strings
    test_strings = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Neural networks are inspired by biological brain structures.",
        "Deep learning has revolutionized computer vision and natural language processing.",
    ]

    # Use the list directly - it already implements __len__ and __getitem__
    dataset = test_strings

    # Load GPT-2 model - use auto device mapping but force concrete tensors
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", device_map="auto", torch_dtype=th.float32
    )
    model = LanguageModel(model, torch_dtype=th.float32, tokenizer=tokenizer)
    model.tokenizer.pad_token = model.tokenizer.eos_token

    # Get a transformer block to extract activations from
    target_layer = model.transformer.h[6]  # Middle layer of GPT-2
    submodule_name = "transformer_h_6"

    # Parameters for activation collection
    batch_size = 2
    context_len = 64
    d_model = 768  # GPT-2 hidden size

    # Collect activations using ActivationCache
    ActivationCache.collect(
        data=dataset,
        submodules=(target_layer,),
        submodule_names=(submodule_name,),
        model=model,
        store_dir=temp_dir,
        batch_size=batch_size,
        context_len=context_len,
        shard_size=1000,  # Small shard size for testing
        d_model=d_model,
        io="out",
        max_total_tokens=10000,
        store_tokens=True,
    )

    # Load the cached activations
    cache = ActivationCache(temp_dir, submodule_name + "_out")

    # Collect activations directly from model for comparison
    direct_activations = []
    direct_tokens = []

    for i in range(0, len(test_strings), batch_size):
        batch_texts = test_strings[i : i + batch_size]

        # Tokenize
        tokens = model.tokenizer(
            batch_texts,
            max_length=context_len,
            truncation=True,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )

        # Get activations directly
        with model.trace(tokens):
            layer_output = target_layer.output[0].save()

        # Extract valid tokens (non-padding)
        attention_mask = tokens["attention_mask"]
        valid_activations = layer_output.reshape(-1, d_model)[
            attention_mask.reshape(-1).bool()
        ]
        valid_tokens = tokens["input_ids"].reshape(-1)[
            attention_mask.reshape(-1).bool()
        ]

        direct_activations.append(valid_activations.cpu())
        direct_tokens.append(valid_tokens.cpu())

    # Concatenate direct activations
    direct_activations = th.cat(direct_activations, dim=0)
    direct_tokens = th.cat(direct_tokens, dim=0)

    # Test that we have the same number of activations
    assert (
        len(cache) == direct_activations.shape[0]
    ), f"Cache length {len(cache)} != direct activations length {direct_activations.shape[0]}"

    # Test that tokens match
    assert th.equal(
        cache.tokens, direct_tokens
    ), "Cached tokens don't match direct tokens"

    # Test that activations match (within tolerance for numerical precision)
    cached_activations = th.stack([cache[i] for i in range(len(cache))], dim=0)
    assert th.allclose(
        cached_activations, direct_activations, atol=1e-5, rtol=1e-5
    ), "Cached activations don't match direct activations"

    # Test mean and std computation
    computed_mean = direct_activations.mean(dim=0)
    computed_std = direct_activations.std(dim=0, unbiased=True)

    assert th.allclose(
        cache.mean, computed_mean, atol=1e-5, rtol=1e-5
    ), "Cached mean doesn't match computed mean"
    assert th.allclose(
        cache.std, computed_std, atol=1e-5, rtol=1e-5
    ), "Cached std doesn't match computed std"

    # Test normalizer functionality
    normalizer = cache.normalizer

    # Test normalization of a sample activation
    sample_activation = cached_activations[0]
    normalized = normalizer(sample_activation)

    # Verify normalization: (x - mean) / std (with small epsilon for numerical stability)
    expected_normalized = (sample_activation - cache.mean) / (cache.std + 1e-8)
    assert th.allclose(
        normalized, expected_normalized, atol=1e-6
    ), "Normalizer doesn't work correctly"

    # Test batch normalization
    batch_normalized = normalizer(cached_activations[:5])
    expected_batch_normalized = (cached_activations[:5] - cache.mean) / (
        cache.std + 1e-8
    )
    assert th.allclose(
        batch_normalized, expected_batch_normalized, atol=1e-6
    ), "Batch normalization doesn't work correctly"

    # Test that normalization preserves shape
    assert (
        normalized.shape == sample_activation.shape
    ), "Normalization changed tensor shape"
    assert (
        batch_normalized.shape == cached_activations[:5].shape
    ), "Batch normalization changed tensor shape"

    print(f"✓ Successfully tested ActivationCache with {len(cache)} activations")
    print(f"✓ Mean shape: {cache.mean.shape}, Std shape: {cache.std.shape}")
    print(f"✓ Normalizer tests passed")
