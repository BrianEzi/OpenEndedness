''' a stripped-down version of the Seer used for pre-training.
The full Seer encodes observations into a discrete FSQ message for the Doer.
Here, I replaced the FSQ head with a simple action head 
so the seer can be trained alone to solve the navigation task.
Once trained, its LSTM weights are reused in the full Seer+Doer system. '''

import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Tuple

class SeerSolo(nn.Module):
    lstm_features: int = 128
    num_actions: int = 3

    @nn.compact
    def __call__(
        self,
        carry: Tuple[jnp.ndarray, jnp.ndarray],
        map_obs: jnp.ndarray,
        symbolic_obs: jnp.ndarray,
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:

        # Visual encoder
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(map_obs)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        x_flat = x.reshape((x.shape[0], -1))

        # Symbolic encoder
        y = nn.Dense(features=64)(symbolic_obs)
        y = nn.relu(y)

        # Fusion
        fused_features = jnp.concatenate([x_flat, y], axis=-1)

        # LSTM
        lstm_cell = nn.LSTMCell(features=self.lstm_features)
        new_carry, lstm_out = lstm_cell(carry, fused_features)

        # Action head
        action_logits = nn.Dense(features=self.num_actions)(lstm_out)

        return new_carry, action_logits

    @staticmethod
    def initialize_carry(batch_size: int, hidden_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return (
            jnp.zeros((batch_size, hidden_size)),
            jnp.zeros((batch_size, hidden_size)),
        )
