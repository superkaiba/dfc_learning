# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository implements dictionary learning via sparse autoencoders (SAEs) on neural network activations. It extends the original dictionary_learning repository with:
- CrossCoders for comparing model variants (e.g., base vs instruct models)
- BatchTopK sparsity mechanisms
- Pip installability and HuggingFace Hub integration
- Activation caching for efficient training

The package is designed for mechanistic interpretability research, particularly for discovering and analyzing features in language models.

## Installation & Setup

```bash
# Development installation
pip install -e .

# Or from GitHub
pip install git+https://github.com/jkminder/dictionary_learning

# Install dependencies
pip install -r requirements.txt
```

The package uses `setuptools_scm` for versioning - version is automatically determined from git tags.

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_cache.py
pytest tests/test_running_stat_welford.py

# Run DFC tests (comprehensive validation of Dedicated Feature Crosscoders)
pytest tests/test_dfc.py -v  # 19 tests including gradient masking and auxiliary loss isolation

# Quick DFC verification (faster, no pytest required)
python test_dfc_quick.py  # 8 critical tests
```

Tests are located in the `tests/` directory. The DFC implementation includes comprehensive tests that validate gradient masking prevents auxiliary loss leakage - a critical requirement for model diffing.

## Code Architecture

### Core Components

**Dictionary Classes** (`dictionary_learning/dictionary.py`):
- `AutoEncoder`: Standard SAE with ReLU activation
- `GatedAutoEncoder`: Gated SAE with separate gating and magnitude networks
- `JumpReluAutoEncoder`: JumpReLU SAE (can load from `sae_lens`)
- `BatchTopKSAE`: Batch-wise top-k sparsity (k features across entire batch)
- `CrossCoder`: Multi-layer SAE for comparing model variants
- `BatchTopKCrossCoder`: CrossCoder with BatchTopK sparsity

All dictionary classes inherit from `Dictionary` base class and support:
- `encode(x)`: Map activations to features
- `decode(f)`: Map features to reconstructed activations
- `forward(x, output_features=False)`: Full forward pass
- `from_pretrained(path, from_hub=False)`: Load from file or HuggingFace Hub
- `push_to_hub(repo_id)`: Save to HuggingFace Hub

**Activation Buffers** (`dictionary_learning/buffer.py`):
- `ActivationBuffer`: Standard buffer for single-layer activations
- `HeadActivationBuffer`: Specialized for attention head activations
- `NNsightActivationBuffer`: Alternative buffer implementation

Buffers handle:
- Extracting activations from models via `nnsight`
- Maintaining a buffer that refreshes when half-depleted
- Yielding batches for training

**Trainers** (`dictionary_learning/trainers/`):
- `StandardTrainer`: Basic SAE training (Bricken et al., 2023)
- `GatedSAETrainer`: Gated SAE training (Rajamanoharan et al., 2024)
- `TrainerTopK`: Top-K SAE training (Gao et al., 2024)
- `PAnnealTrainer`: Standard trainer with p-annealing
- `GatedAnnealTrainer`: Gated trainer with p-annealing
- `TrainerJumpRelu`: JumpReLU SAE training
- `BatchTopKTrainer`: BatchTopK SAE training
- `CrossCoderTrainer`: CrossCoder training
- `BatchTopKCrossCoderTrainer`: BatchTopK CrossCoder training

Each trainer implements `loss()` method and manages training-specific logic.

**Training Infrastructure** (`dictionary_learning/training.py`):
- `trainSAE()`: Main training loop supporting multiple trainers
- `ConstrainedAdam`: Adam optimizer with decoder weight norm constraints
- Neuron resampling (via `resample_steps` parameter)
- Learning rate warmup (via `warmup_steps` parameter)
- Weights & Biases logging integration

### CrossCoder Architecture

CrossCoders process activations from multiple layers simultaneously:

1. **Encoder** (`CrossCoderEncoder`): Takes multi-layer activations `(batch, n_layers, d_model)`, applies layer-specific linear transformations, sums across layers, adds bias, applies ReLU
2. **Decoder** (`CrossCoderDecoder`): Takes dictionary features `(batch, dict_size)`, applies layer-specific linear transformations to reconstruct `(batch, n_layers, d_model)`
3. **Code Normalization** (`CodeNormalization` enum): Controls how features are scaled:
   - `CROSSCODER`: Sum of decoder weight norms across layers (encourages multi-layer features)
   - `SAE`: Norm of concatenated weights (treats layers equally)
   - `MIXED`: Weighted combination of SAE and CROSSCODER
   - `DECOUPLED`: Per-layer normalization
   - `NONE`: No normalization

BatchTopKCrossCoder adds:
- Adaptive thresholding based on top-k selection during training
- Auxiliary loss for dead feature resurrection
- Optional k-annealing (gradually reduce k during training)

### Dedicated Feature Crosscoders (DFCs)

**NEW:** DFCs extend the CrossCoder architecture with feature partitioning for improved model diffing.

**What are DFCs?**
Dedicated Feature Crosscoders partition the feature dictionary into three disjoint sets:
- **Model A-exclusive features**: Only Model A can encode/decode from these
- **Model B-exclusive features**: Only Model B can encode/decode from these
- **Shared features**: Both models can encode/decode from these

This architectural constraint creates a prior favoring model-exclusive feature discovery, which is valuable for model diffing applications where the goal is to identify behavioral differences.

**Key Classes** (`dictionary_learning/dictionary.py`):
- `FeaturePartition`: Manages partition indices and masks
- `DedicatedFeatureCrossCoderEncoder`: Encoder with gradient masking
- `DedicatedFeatureCrossCoderDecoder`: Decoder with gradient masking
- `DedicatedFeatureCrossCoder`: Base DFC class
- `DedicatedFeatureBatchTopKCrossCoder`: DFC with BatchTopK sparsity (recommended)

**How DFCs Work:**
1. **Initialization**: Forbidden weights (e.g., Model A's decoder for B-exclusive features) are set to zero
2. **Gradient Masking**: PyTorch hooks zero out gradients for forbidden parameters during backprop
3. **Optimizer Safety**: Momentum terms (exp_avg, exp_avg_sq) are zeroed for forbidden parameters
4. **Integrity Monitoring**: `verify_partition_integrity()` checks that forbidden weights remain zero

**Critical Feature: Auxiliary Loss Protection**
The gradient masking prevents auxiliary loss (used for dead feature resurrection) from leaking gradients to the wrong model's exclusive features. This is verified by comprehensive tests.

**Partition Configuration:**
Partitions are specified as percentages of total features:
```python
dfc = DedicatedFeatureBatchTopKCrossCoder(
    activation_dim=2304,
    dict_size=131072,
    num_layers=2,
    k=200,
    model_a_exclusive_pct=0.05,  # 5% = 6,554 features for Model A
    model_b_exclusive_pct=0.05,  # 5% = 6,554 features for Model B
    # Remaining 90% = 117,964 shared features
)
```

**Identifying Model-Exclusive Features:**
After training, model-exclusive features can be identified by:
1. **Partition membership**: Features in dedicated partitions (by design)
2. **Relative decoder norm**: `||d_A|| / (||d_A|| + ||d_B||)` (near 0 or 1 for exclusive features)
3. **Activation steering**: Steering should only affect the designated model
4. **Transfer testing**: Features shouldn't transfer via learned affine maps

**Training DFCs:**
DFCs work with the existing `BatchTopKCrossCoderTrainer` without modification:
```python
from dictionary_learning import DedicatedFeatureBatchTopKCrossCoder
from dictionary_learning.trainers.crosscoder import BatchTopKCrossCoderTrainer

dfc = DedicatedFeatureBatchTopKCrossCoder(
    activation_dim=2304,
    dict_size=131072,
    num_layers=2,
    k=200,
    model_a_exclusive_pct=0.05,
    model_b_exclusive_pct=0.05,
)

trainer = BatchTopKCrossCoderTrainer(
    steps=100000,
    activation_dim=2304,
    dict_size=131072,
    k=200,
    layer=13,
    lm_name="model-a_vs_model-b",
    lr=1e-4,
    dict_class=DedicatedFeatureBatchTopKCrossCoder,
    dict_class_kwargs={
        "model_a_exclusive_pct": 0.05,
        "model_b_exclusive_pct": 0.05,
    },
)
```

**Monitoring Training:**
The trainer automatically logs partition integrity metrics:
- `partition_integrity_encoder_max`: Max violation for encoder weights
- `partition_integrity_decoder_max`: Max violation for decoder weights
- Values should stay near zero (< 1e-6) throughout training

**Testing DFCs:**
Comprehensive tests validate the implementation:
```bash
# Quick verification (8 critical tests)
python test_dfc_quick.py

# Full test suite (19 tests including auxiliary loss isolation)
pytest tests/test_dfc.py -v
```

**Partition Size Considerations:**
- **Smaller partitions (1-2%)**: Better shared feature quality, may miss fine-grained differences
- **Larger partitions (5-10%)**: Captures more model-specific features, trades off shared capacity
- **Recommended start**: 5% for each model (paper's default)

**Important Notes:**
- DFCs are designed for **cross-architecture model diffing** (different models, not base vs finetune)
- Gradient masking is essential - without it, auxiliary loss would leak gradients
- Partition integrity should be monitored during training (automatically logged)
- Features in exclusive partitions have **architectural guarantees** of exclusivity
- Save/load automatically preserves partition configuration

### Activation Caching

**Activation Cache** (`dictionary_learning/cache.py`):
- `PairedActivationCache`: Stores paired activations from two models (e.g., base vs instruct)
- Used by `scripts/train_crosscoder.py` to train CrossCoders offline
- Enables efficient training without re-running model forward passes

Cache format: Activations stored in directories organized by model/dataset/layer.

## Common Development Workflows

### Training a Standard SAE

```python
from nnsight import LanguageModel
from dictionary_learning import ActivationBuffer, AutoEncoder
from dictionary_learning.trainers import StandardTrainer
from dictionary_learning.training import trainSAE

model = LanguageModel("EleutherAI/pythia-70m-deduped", device_map="cuda:0")
submodule = model.gpt_neox.layers[1].mlp
activation_dim = 512
dictionary_size = 16 * activation_dim

# Create data iterator
data = iter(["text sample 1", "text sample 2", ...])

buffer = ActivationBuffer(
    data=data,
    model=model,
    submodule=submodule,
    d_submodule=activation_dim,
    n_ctxs=3e4,
    device="cuda:0"
)

trainer_cfg = {
    "trainer": StandardTrainer,
    "dict_class": AutoEncoder,
    "activation_dim": activation_dim,
    "dict_size": dictionary_size,
    "lr": 1e-3,
    "device": "cuda:0",
}

ae = trainSAE(
    data=buffer,
    trainer_configs=[trainer_cfg],
    resample_steps=25000,  # Resample dead neurons every 25k steps
    warmup_steps=1000,     # LR warmup
)
```

### Training a CrossCoder with Pre-computed Activations

```bash
python dictionary_learning/scripts/train_crosscoder.py \
    --activation-store-dir activations \
    --base-model gemma-2-2b \
    --instruct-model gemma-2-2b-it \
    --layer 13 \
    --expansion-factor 32 \
    --mu 1e-1 \
    --lr 1e-3
```

Activations should be cached in: `activations/<model>/<dataset>/<submodule>/`

### Loading and Using Pretrained Dictionaries

```python
from dictionary_learning import AutoEncoder, CrossCoder

# Load from local file
ae = AutoEncoder.from_pretrained("path/to/ae.pt")

# Load from HuggingFace Hub
crosscoder = CrossCoder.from_pretrained(
    "Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04",
    from_hub=True
)

# Use dictionary
activations = torch.randn(64, activation_dim)
features = ae.encode(activations)
reconstructed = ae.decode(features)

# Or combined
reconstructed, features = ae(activations, output_features=True)
```

### Adding a New Trainer

1. Create file in `dictionary_learning/trainers/`
2. Inherit from `Trainer` base class (`trainers/trainer.py`)
3. Implement `__init__`, `loss()`, and optionally `get_logging_parameters()`
4. Add import to `trainers/__init__.py`
5. Add corresponding dictionary class if needed in `dictionary.py`

The `loss()` method should return `(x, x_hat, f, losslog)` where:
- `x`: Input activations
- `x_hat`: Reconstructed activations
- `f`: Dictionary features
- `losslog`: Dict of logged values (should include "loss" key)

## Key Dependencies

- **nnsight** (>=0.2.11): Extract and intervene on model activations
- **torch** (>=2.1.0): Core tensor operations
- **einops** (>=0.7.0): Tensor reshaping operations
- **datasets** (>=2.18.0): Dataset loading
- **wandb**: Experiment tracking (optional but recommended)
- **huggingface_hub**: Model sharing via `PyTorchModelHubMixin`

## Code Formatting

The project uses `black` for code formatting:
```bash
# Format code
black dictionary_learning/

# Pre-commit hook available
pre-commit install
```

Configuration: `.pre-commit-config.yaml`

## Important Notes

- The package uses `nnsight` for activation extraction - if `nnsight` has breaking changes, buffer implementations may need updates
- When working with residual stream activations (tuples), buffers automatically extract the first element
- Decoder weights can be constrained to unit norm using `ConstrainedAdam` optimizer
- Ghost grads are supported in `AutoEncoder` via the `ghost_mask` parameter in `forward()`
- Dead neuron resampling follows the procedure from "Towards Monosemanticity" (Anthropic, 2023)
- CrossCoders use layer-wise operations - be careful with tensor dimensions `(batch, n_layers, d_model)`

## Experimental Features

The following features are supported but may be deprecated:
- MLP stretchers (mapping MLP input → output via dictionary)
- Entropy-based sparsity regularization (instead of L1)
- Ghost grads for dead feature resurrection
