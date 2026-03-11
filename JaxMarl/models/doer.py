import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Tuple

class Doer(nn.Module):
    """
    The 'Thief' or Motor Cortex network.
    Operates on local observations and executes commands via discrete actions.
    """
    fsq_levels: Sequence[int]
    num_actions: int = 6 # e.g., Move N/S/E/W, Toggle, Pick Up
    lstm_features: int = 128
    embed_dim: int = 16

    @nn.compact
    def __call__(
        self,
        carry: Tuple[jnp.ndarray, jnp.ndarray],
        local_obs: jnp.ndarray,
        proprioception: jnp.ndarray,
        message: jnp.ndarray
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """
        Args:
            carry: A tuple of (hidden_state, cell_state) for the LSTM.
            local_obs: Egocentric 3x3 grid view. Expected shape (batch, 3, 3, C) or zeros.
            proprioception: Internal states (e.g., carrying item). Expected shape (batch, features).
            message: The quantized $m_t$ vector from the Seer. Expected shape (batch, d).
            
        Returns:
            new_carry: Updated LSTM state for the next timestep $t+1$.
            action_logits: Unnormalized log probabilities for the discrete action space.
        """
        
        # 1. Message Encoder: Learned Lookup Table
        # The message is a vector of length d (e.g., [1.0, 4.0, 0.0]). 
        # We cast to integer and embed each dimension separately, then flatten.
        message_int = message.astype(jnp.int32)
        embedded_dims = []
        
        for i, num_levels in enumerate(self.fsq_levels):
            # Each dimension of the FSQ message gets its own embedding space
            # This respects the factored nature of the quantized vector
            emb = nn.Embed(num_embeddings=num_levels, features=self.embed_dim)
            # Extract the i-th dimension across the batch
            embedded_dims.append(emb(message_int[:, i]))
            
        # Concatenate all embedded dimensions into a single vector per batch item
        message_features = jnp.concatenate(embedded_dims, axis=-1)
        
        # 2. Local Visual Encoder
        # Even for a small 3x3 grid, a single convolution or a dense layer extracts features
        x = nn.Conv(features=16, kernel_size=(2, 2))(local_obs)
        x = nn.relu(x)
        x_flat = x.reshape((x.shape[0], -1))
        
        # 3. Proprioception Encoder
        p = nn.Dense(features=16)(proprioception)
        p = nn.relu(p)
        
        # 4. Fusion
        # Combine local vision, proprioception, and the embedded command
        fused_features = jnp.concatenate([x_flat, p, message_features], axis=-1)
        
        # 5. Reasoning Module: LSTM
        # Critical for integrating sequences of commands (e.g., maintaining a "Wait" state)
        lstm_cell = nn.LSTMCell(features=self.lstm_features)
        new_carry, lstm_out = lstm_cell(carry, fused_features)
        
        # 6. Output Head: Discrete Action Space
        # Projects the LSTM memory state into logits for physical actions
        action_logits = nn.Dense(features=self.num_actions)(lstm_out)
        
        return new_carry, action_logits

    @staticmethod
    def initialize_carry(batch_size: int, hidden_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Utility to generate the initial zero-state for the LSTM at the start of an episode."""
        return (
            jnp.zeros((batch_size, hidden_size)), 
            jnp.zeros((batch_size, hidden_size))
        )