"""Feature partition management for Dedicated Feature Crosscoders (DFCs).

This module implements the partition logic that divides a dictionary's feature
space into model-exclusive and shared subspaces, enabling architectural constraints
for model diffing applications.
"""

import torch as th
from typing import Optional


class FeaturePartition:
    """Manages feature partitioning for Dedicated Feature Crosscoders.

    The partition divides a dictionary of size `dict_size` into three disjoint sets:
    1. Model A exclusive features: indices [0, model_a_exclusive_size)
    2. Model B exclusive features: indices [model_a_exclusive_size, model_a_exclusive_size + model_b_exclusive_size)
    3. Shared features: indices [model_a_exclusive_size + model_b_exclusive_size, dict_size)

    The partition enforces architectural constraints:
    - Model A (layer 0) can only encode/decode from A-exclusive + shared features
    - Model B (layer 1) can only encode/decode from B-exclusive + shared features

    This prevents gradient flow to forbidden features and maintains strict partitioning
    throughout training.

    Args:
        dict_size: Total number of features in the dictionary
        model_a_exclusive_size: Number of features exclusive to Model A
        model_b_exclusive_size: Number of features exclusive to Model B

    Example:
        >>> partition = FeaturePartition(dict_size=1000, model_a_exclusive_size=50, model_b_exclusive_size=50)
        >>> partition.model_a_indices  # tensor([0, 1, ..., 49])
        >>> partition.shared_indices   # tensor([100, 101, ..., 999])
        >>> mask = partition.get_encoder_mask(layer_idx=0)  # For Model A
        >>> # mask has 0s for Model B exclusive features, 1s elsewhere
    """

    def __init__(
        self,
        dict_size: int,
        model_a_exclusive_size: int,
        model_b_exclusive_size: int,
    ):
        """Initialize feature partition.

        Args:
            dict_size: Total number of features in the dictionary
            model_a_exclusive_size: Number of features exclusive to Model A
            model_b_exclusive_size: Number of features exclusive to Model B

        Raises:
            ValueError: If partition sizes exceed dict_size or are negative
        """
        if model_a_exclusive_size < 0 or model_b_exclusive_size < 0:
            raise ValueError("Exclusive sizes must be non-negative")

        if model_a_exclusive_size + model_b_exclusive_size > dict_size:
            raise ValueError(
                f"Exclusive sizes sum ({model_a_exclusive_size + model_b_exclusive_size}) "
                f"exceeds dict_size ({dict_size})"
            )

        self.dict_size = dict_size
        self.model_a_exclusive_size = model_a_exclusive_size
        self.model_b_exclusive_size = model_b_exclusive_size
        self.shared_size = dict_size - model_a_exclusive_size - model_b_exclusive_size

        # Precompute index ranges (as 1D tensors)
        self._model_a_indices = th.arange(0, model_a_exclusive_size)
        self._model_b_indices = th.arange(
            model_a_exclusive_size,
            model_a_exclusive_size + model_b_exclusive_size
        )
        self._shared_indices = th.arange(
            model_a_exclusive_size + model_b_exclusive_size,
            dict_size
        )

        # Precompute allowed indices (concatenated ranges)
        self._model_a_allowed_indices = th.cat([self._model_a_indices, self._shared_indices])
        self._model_b_allowed_indices = th.cat([self._model_b_indices, self._shared_indices])

    @classmethod
    def from_percentages(
        cls,
        dict_size: int,
        model_a_exclusive_pct: float,
        model_b_exclusive_pct: float,
    ) -> "FeaturePartition":
        """Create partition from percentage specifications.

        Args:
            dict_size: Total number of features in the dictionary
            model_a_exclusive_pct: Percentage of features for Model A (e.g., 0.05 for 5%)
            model_b_exclusive_pct: Percentage of features for Model B (e.g., 0.05 for 5%)

        Returns:
            FeaturePartition instance with computed sizes

        Example:
            >>> partition = FeaturePartition.from_percentages(1000, 0.05, 0.05)
            >>> partition.model_a_exclusive_size  # 50
            >>> partition.shared_size             # 900
        """
        model_a_size = int(dict_size * model_a_exclusive_pct)
        model_b_size = int(dict_size * model_b_exclusive_pct)
        return cls(dict_size, model_a_size, model_b_size)

    @property
    def model_a_indices(self) -> th.Tensor:
        """Indices of features exclusive to Model A.

        Returns:
            1D tensor of indices [0, model_a_exclusive_size)
        """
        return self._model_a_indices

    @property
    def model_b_indices(self) -> th.Tensor:
        """Indices of features exclusive to Model B.

        Returns:
            1D tensor of indices [model_a_exclusive_size, model_a_exclusive_size + model_b_exclusive_size)
        """
        return self._model_b_indices

    @property
    def shared_indices(self) -> th.Tensor:
        """Indices of features shared between both models.

        Returns:
            1D tensor of indices [model_a_exclusive_size + model_b_exclusive_size, dict_size)
        """
        return self._shared_indices

    @property
    def model_a_allowed_indices(self) -> th.Tensor:
        """Indices of features Model A can encode/decode from.

        Returns:
            1D tensor containing A-exclusive + shared indices
        """
        return self._model_a_allowed_indices

    @property
    def model_b_allowed_indices(self) -> th.Tensor:
        """Indices of features Model B can encode/decode from.

        Returns:
            1D tensor containing B-exclusive + shared indices
        """
        return self._model_b_allowed_indices

    def get_encoder_mask(self, layer_idx: int, device: Optional[th.device] = None) -> th.Tensor:
        """Get encoder weight mask for specified layer.

        The encoder has shape (num_layers, activation_dim, dict_size).
        This returns a mask for the dict_size dimension, with:
        - 1.0 for features this layer can encode to
        - 0.0 for forbidden features

        Args:
            layer_idx: Layer index (0 for Model A, 1 for Model B)
            device: Device to place mask on (default: CPU)

        Returns:
            Tensor of shape (dict_size,) with 1s for allowed features, 0s for forbidden

        Example:
            >>> partition = FeaturePartition(100, 10, 10)
            >>> mask_a = partition.get_encoder_mask(0)
            >>> mask_a[5]   # 1.0 (A-exclusive)
            >>> mask_a[15]  # 0.0 (B-exclusive)
            >>> mask_a[50]  # 1.0 (shared)
        """
        mask = th.zeros(self.dict_size, device=device)

        if layer_idx == 0:  # Model A
            mask[self._model_a_allowed_indices] = 1.0
        elif layer_idx == 1:  # Model B
            mask[self._model_b_allowed_indices] = 1.0
        else:
            raise ValueError(f"Layer index {layer_idx} not supported (expected 0 or 1)")

        return mask

    def get_decoder_mask(self, layer_idx: int, device: Optional[th.device] = None) -> th.Tensor:
        """Get decoder weight mask for specified layer.

        The decoder has shape (num_layers, dict_size, activation_dim).
        This returns a mask for the dict_size dimension, with:
        - 1.0 for features this layer can decode from
        - 0.0 for forbidden features

        Args:
            layer_idx: Layer index (0 for Model A, 1 for Model B)
            device: Device to place mask on (default: CPU)

        Returns:
            Tensor of shape (dict_size,) with 1s for allowed features, 0s for forbidden

        Note:
            Same logic as encoder mask - both models can only access their dedicated + shared features
        """
        # Decoder mask is same as encoder mask (same partition logic)
        return self.get_encoder_mask(layer_idx, device=device)

    def get_bias_mask(self, device: Optional[th.device] = None) -> th.Tensor:
        """Get encoder bias mask.

        The encoder bias has shape (dict_size,).
        All features can have bias since it's shared across layers,
        so this always returns a mask of all ones.

        Args:
            device: Device to place mask on (default: CPU)

        Returns:
            Tensor of shape (dict_size,) with all 1s

        Note:
            Included for API completeness. No masking needed for bias.
        """
        return th.ones(self.dict_size, device=device)

    def to_dict(self) -> dict:
        """Serialize partition configuration for saving.

        Returns:
            Dictionary containing partition configuration
        """
        return {
            "dict_size": self.dict_size,
            "model_a_exclusive_size": self.model_a_exclusive_size,
            "model_b_exclusive_size": self.model_b_exclusive_size,
            "shared_size": self.shared_size,
        }

    @classmethod
    def from_dict(cls, config: dict) -> "FeaturePartition":
        """Deserialize partition configuration from saved state.

        Args:
            config: Dictionary containing partition configuration

        Returns:
            FeaturePartition instance reconstructed from config
        """
        return cls(
            dict_size=config["dict_size"],
            model_a_exclusive_size=config["model_a_exclusive_size"],
            model_b_exclusive_size=config["model_b_exclusive_size"],
        )

    def __repr__(self) -> str:
        """String representation of partition."""
        return (
            f"FeaturePartition(dict_size={self.dict_size}, "
            f"model_a_exclusive={self.model_a_exclusive_size}, "
            f"model_b_exclusive={self.model_b_exclusive_size}, "
            f"shared={self.shared_size})"
        )
