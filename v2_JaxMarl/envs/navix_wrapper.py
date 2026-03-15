import jax
import jax.numpy as jnp
import numpy as np
from navix import observations


class NavixGridWrapper:
    """Expose Navix single-agent environments with the Seer/Doer observation split."""

    def __init__(self, env):
        self._env = env

    @property
    def num_actions(self) -> int:
        return int(self._env.action_space.n)

    def _split_observations(self, timestep, vision_radius: jnp.ndarray):
        state = timestep.state
        global_map = observations.symbolic(state).astype(jnp.float32)
        local_view = observations.symbolic_first_person(state).astype(jnp.float32)

        player = state.get_player()
        goal = state.get_goals()
        symbolic_state = jnp.array(
            [
                player.position[0],
                player.position[1],
                player.direction,
                player.pocket,
                goal.position[0, 0],
                goal.position[0, 1],
            ],
            dtype=jnp.float32,
        )

        # Navix first-person view is centered around the agent; curriculum masks by radius.
        height, width = local_view.shape[:2]
        center_y = height // 2
        center_x = width // 2
        yy, xx = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
        dist = jnp.maximum(jnp.abs(xx - center_x), jnp.abs(yy - center_y))
        mask = (dist <= vision_radius)[..., None]
        local_view = jnp.where(mask, local_view, 0)

        proprioception = jnp.array(
            [
                player.position[0],
                player.position[1],
                player.direction,
                player.pocket,
            ],
            dtype=jnp.float32,
        )

        return {
            "global_map": global_map,
            "symbolic_state": symbolic_state,
            "local_view": local_view,
            "proprioception": proprioception,
        }

    def reset(self, key: jnp.ndarray, vision_radius: jnp.ndarray = jnp.array(3.0)):
        timestep = self._env.reset(key)
        obs = self._split_observations(timestep, vision_radius)
        return obs, timestep

    def reset_batch(self, keys: jnp.ndarray, vision_radius: jnp.ndarray = jnp.array(3.0)):
        return jax.vmap(self.reset, in_axes=(0, None))(keys, vision_radius)

    def step(
        self,
        key: jnp.ndarray,
        timestep,
        action: jnp.ndarray,
        vision_radius: jnp.ndarray = jnp.array(3.0),
    ):
        del key  # Navix carries the RNG inside the timestep state.
        next_timestep = self._env.step(timestep, action)
        obs = self._split_observations(next_timestep, vision_radius)
        reward = next_timestep.reward
        done = next_timestep.is_done()
        return obs, next_timestep, reward, done, next_timestep.info

    def step_batch(
        self,
        keys: jnp.ndarray,
        timesteps,
        actions: jnp.ndarray,
        vision_radius: jnp.ndarray = jnp.array(3.0),
    ):
        return jax.vmap(self.step, in_axes=(0, 0, 0, None))(keys, timesteps, actions, vision_radius)

    def render(self, timestep):
        return np.asarray(observations.rgb(timestep.state))
