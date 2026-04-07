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
    num_actions: int = 3
    lstm_features: int = 128
    num_message_heads: int = 1

    @nn.compact
    def __call__(
        self, 
        carry: Tuple[jnp.ndarray, jnp.ndarray], 
        map_obs: jnp.ndarray, 
        symbolic_obs: jnp.ndarray,
        target_images: jnp.ndarray
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
            carry: A tuple of (hidden_state, cell_state) for the LSTM.
            map_obs: The Global Map View grid. Expected shape (batch, H, W, C).
            symbolic_obs: Guard Schedule + Sensor States. Expected shape (batch, features).
            target_images: The items the Doers must select. Expected shape (batch, 2, 5, 5, 3).
            
        Returns:
            new_carry: Updated LSTM state for the next timestep $t+1$.
            discrete_message: The quantized $m_t$ vector sent to the Doer.
            thought_vector: The continuous pre-quantization vector.
            navigation_logits: Action logits used while the Seer is embodied.
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
        
        # 3. Target Images Encoder
        # target_images shape is (batch, 2, 5, 5, 3)
        ti_flat = target_images.reshape((-1, 5, 5, 3))
        ti_conv1 = nn.Conv(features=16, kernel_size=(3, 3))(ti_flat)
        ti_conv1 = nn.relu(ti_conv1)
        ti_conv2 = nn.Conv(features=32, kernel_size=(3, 3))(ti_conv1)
        ti_conv2 = nn.relu(ti_conv2)
        ti_feats = ti_conv2.reshape((target_images.shape[0], -1))
        
        # 4. Fusion
        # Concatenate the visual, symbolic, and target features
        fused_features = jnp.concatenate([x_flat, y, ti_feats], axis=-1)
        
        # 5. Reasoning Module: LSTM 
        # Evaluates the current state in the context of the previous timestep [cite: 135]
        lstm_cell = nn.LSTMCell(features=self.lstm_features)
        new_carry, lstm_out = lstm_cell(carry, fused_features)
        
        # 5. Continuous Projection
        # Project LSTM hidden state to continuous vector z of size d.
        # Use Orthogonal init with higher scale to prevent FSQ mode collapse.
        thought_vector = nn.Dense(
            features=len(self.fsq_levels) * self.num_message_heads,
            kernel_init=nn.initializers.orthogonal(scale=2.0)
        )(lstm_out)
        thought_vector = thought_vector.reshape(
            (thought_vector.shape[0], self.num_message_heads, len(self.fsq_levels))
        )

        # 6. Output Head: FSQ Discretizer 
        # Transforms the continuous thought vector into the discrete message $m_t$ 
        fsq = FSQ(levels=self.fsq_levels)
        discrete_message = fsq(
            thought_vector.reshape((-1, thought_vector.shape[-1]))
        ).reshape(thought_vector.shape)

        if self.num_message_heads == 1:
            thought_vector = thought_vector[:, 0, :]
            discrete_message = discrete_message[:, 0, :]

        # During the pretraining phase the Seer physically navigates the grid.
        navigation_logits = nn.Dense(features=self.num_actions)(lstm_out)

        return new_carry, discrete_message, thought_vector, navigation_logits

    @staticmethod
    def initialize_carry(batch_size: int, hidden_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Utility to generate the initial zero-state for the LSTM at the start of an episode."""
        return (
            jnp.zeros((batch_size, hidden_size)), 
            jnp.zeros((batch_size, hidden_size))
        )
