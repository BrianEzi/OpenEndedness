import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Tuple

# Import the FSQ module we defined previously
from models.fsq import FSQ

class Seer(nn.Module):
    """
    The 'Hacker' or Prefrontal Cortex network.
    Observes the global state and generates a discrete compositional message.
    """
    fsq_levels: Sequence[int]
    lstm_features: int = 128

    @nn.compact
    def __call__(
        self, 
        carry: Tuple[jnp.ndarray, jnp.ndarray], 
        map_obs: jnp.ndarray, 
        symbolic_obs: jnp.ndarray
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
        """
        Args:
            carry: A tuple of (hidden_state, cell_state) for the LSTM.
            map_obs: The Global Map View grid. Expected shape (batch, H, W, C).
            symbolic_obs: Guard Schedule + Sensor States. Expected shape (batch, features).
            
        Returns:
            new_carry: Updated LSTM state for the next timestep $t+1$.
            discrete_message: The quantized $m_t$ vector sent to the Doer.
            thought_vector: The continuous pre-quantization vector (useful for logging/critic).
        """
        
        # 1. Visual Encoder: CNN for the grid visual 
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(map_obs)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        
        # Flatten visual features to a 1D vector per batch item
        # shape changes from (batch, H, W, channels) to (batch, H * W * channels)
        x_flat = x.reshape((x.shape[0], -1)) 
        
        # 2. Symbolic Encoder: MLP for symbolic data 
        y = nn.Dense(features=64)(symbolic_obs)
        y = nn.relu(y)
        
        # 3. Fusion
        # Concatenate the visual and symbolic pathways into a single representation 
        fused_features = jnp.concatenate([x_flat, y], axis=-1)
        
        # 4. Reasoning Module: LSTM 
        # Evaluates the current state in the context of the previous timestep [cite: 135]
        lstm_cell = nn.LSTMCell(features=self.lstm_features)
        new_carry, lstm_out = lstm_cell(carry, fused_features)
        
        # 5. Continuous Projection
        # Project the LSTM output to the exact number of dimensions (d) required by FSQ
        d = len(self.fsq_levels)
        thought_vector = nn.Dense(features=d)(lstm_out)
        
        # 6. Output Head: FSQ Discretizer 
        # Transforms the continuous thought vector into the discrete message $m_t$ 
        fsq = FSQ(levels=self.fsq_levels)
        discrete_message = fsq(thought_vector)
        
        return new_carry, discrete_message, thought_vector

    @staticmethod
    def initialize_carry(batch_size: int, hidden_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Utility to generate the initial zero-state for the LSTM at the start of an episode."""
        return (
            jnp.zeros((batch_size, hidden_size)), 
            jnp.zeros((batch_size, hidden_size))
        )