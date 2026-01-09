"""
APEXNEURO - SPIKE ENCODER (SOFT HDC)
Holographic projection using Tanh activation to preserve signal magnitude.
"""

import jax
import jax.numpy as jnp
import chex

@chex.dataclass
class SpikeHDCEncoder:
    channel_basis: jnp.ndarray  # Shape: (num_channels, hyper_dim)
    hyper_dim: int
    num_channels: int

    @classmethod
    def create(cls, key: jax.random.PRNGKey, num_channels: int, hyper_dim: int = 10000):
        # Initialize random projection matrix (Fixed weights)
        # We scale by 1/sqrt(N) to keep values stable
        raw_basis = jax.random.normal(key, shape=(num_channels, hyper_dim))
        channel_basis = raw_basis / jnp.sqrt(num_channels)
        
        return cls(
            channel_basis=channel_basis,
            hyper_dim=hyper_dim,
            num_channels=num_channels
        )

    @jax.jit
    def encode_batch(self, spike_matrix: jnp.ndarray) -> jnp.ndarray:
        """
        Projects spikes into high-dimensional space with a soft nonlinearity.
        """
        # 1. Random Projection (Mixing)
        superposition = jnp.dot(spike_matrix, self.channel_basis)
        
        # 2. Nonlinearity (Reservoir Mode)
        # Tanh preserves the 'analog' velocity info better than Sign
        return jnp.tanh(superposition)