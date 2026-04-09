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
    num_actions: int = 9 # e.g., Move N/S/E/W, Toggle, Pick 0/1/2/3
    lstm_features: int = 128
    embed_dim: int = 16

    @nn.compact
    def __call__(
        self,
        carry: Tuple[jnp.ndarray, jnp.ndarray],
        local_obs: jnp.ndarray,
        proprioception: jnp.ndarray,
        message: jnp.ndarray,
        menu_images: jnp.ndarray
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """
        Args:
            carry: A tuple of (hidden_state, cell_state) for the LSTM.
            local_obs: Egocentric 3x3 grid view. Expected shape (batch, 3, 3, C) or zeros.
            proprioception: Internal states (e.g., carrying item). Expected shape (batch, features).
            message: The quantized $m_t$ vector from the Seer. Expected shape (batch, d).
            menu_images: The 4 option images. Expected shape (batch, 4, 5, 5, 3).
            
        Returns:
            new_carry: Updated LSTM state for the next timestep $t+1$.
            action_logits: Unnormalized log probabilities for the discrete action space.
        """
        
        # 1. Message Encoder: Learned Lookup Table
        # To preserve gradients, we MUST NOT cast to int and use nn.Embed directly 
        # Instead, FSQ naturally provides continuous coordinates (that happen to be quantized).
        # We can just linearly project this entire vector directly into a latent space!
        message_features = nn.Dense(features=self.embed_dim * len(self.fsq_levels))(message)
        message_features = nn.relu(message_features)
        
        # 2. Local Visual Encoder
        # Even for a small 3x3 grid, a single convolution or a dense layer extracts features
        x = nn.Conv(features=16, kernel_size=(2, 2))(local_obs)
        x = nn.relu(x)
        x_flat = x.reshape((x.shape[0], -1))
        
        # 3. Proprioception Encoder
        p = nn.Dense(features=16)(proprioception)
        p = nn.relu(p)
        
        # 4. Menu Images Encoder
        # menu_images shape is (batch, 4, 5, 5, 3)
        mi_flat = menu_images.reshape((-1, 5, 5, 3))
        mi_conv1 = nn.Conv(features=16, kernel_size=(3, 3))(mi_flat)
        mi_conv1 = nn.relu(mi_conv1)
        mi_conv2 = nn.Conv(features=32, kernel_size=(3, 3))(mi_conv1)
        mi_conv2 = nn.relu(mi_conv2)
        mi_feats = mi_conv2.reshape((menu_images.shape[0], -1))
        
        # 5. Fusion
        # Combine local vision, proprioception, menu images, and the embedded command
        fused_features = jnp.concatenate([x_flat, p, mi_feats, message_features], axis=-1)
        
        # 6. Reasoning Module: LSTM
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