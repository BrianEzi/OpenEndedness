import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax

class ActorCritic(nn.Module):
    @nn.compact
    def __call__(self, obs):
        # Seer Head
        # Input: obs['seer'] -> (batch, 4) flattened [doer_pos, target_pos]
        seer_x = nn.Dense(64)(obs['seer'])
        seer_x = nn.relu(seer_x)
        seer_x = nn.Dense(64)(seer_x)
        seer_x = nn.relu(seer_x)
        seer_mean = nn.Dense(5)(seer_x)
        # Using a fixed standard deviation of 0.5 effectively makes it a deterministic communication 
        # plus gaussian noise channel.
        # Ensure scale is broadcastable if needed, but ones_like handles batching.
        seer_dist = distrax.MultivariateNormalDiag(
            loc=seer_mean, 
            scale_diag=jnp.ones_like(seer_mean) * 0.5
        )

        # Doer Head
        # Input: obs['doer'] -> (batch, 5) last_message
        doer_x = nn.Dense(64)(obs['doer'])
        doer_x = nn.relu(doer_x)
        doer_x = nn.Dense(64)(doer_x)
        doer_x = nn.relu(doer_x)
        doer_logits = nn.Dense(5)(doer_x) # 5 possible moves
        doer_dist = distrax.Categorical(logits=doer_logits)

        # Critic
        # Centralized Value Function: takes both observations
        critic_in = jnp.concatenate([obs['seer'], obs['doer']], axis=-1)
        v = nn.Dense(64)(critic_in)
        v = nn.relu(v)
        v = nn.Dense(64)(v)
        v = nn.relu(v)
        value = nn.Dense(1)(v).squeeze(-1)

        return seer_dist, doer_dist, value
