import jax
import jax.numpy as jnp
import numpy as np
from navix import observations


class NavixGridWrapper:
    """Expose Navix single-agent environments with the Seer/Doer observation split."""

    def __init__(
        self,
        env,
        progress_reward_scale: float = 0.1,
        min_start_distance: float = 0.0,
        step_penalty: float = 0.0,
    ):
        self._env = env
        self.progress_reward_scale = progress_reward_scale
        self.min_start_distance = jnp.asarray(min_start_distance, dtype=jnp.float32)
        self.step_penalty = jnp.asarray(step_penalty, dtype=jnp.float32)

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

        # The Doer is fully blind: no visual grid is exposed at all.
        local_view = jnp.zeros_like(local_view)

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

    @staticmethod
    def _goal_distance(state) -> jnp.ndarray:
        player = state.get_player()
        goal = state.get_goals()
        return jnp.abs(player.position - goal.position[0]).sum().astype(jnp.float32)

    def reset(self, key: jnp.ndarray, vision_radius: jnp.ndarray = jnp.array(3.0)):
        timestep = self._env.reset(key)
        distance = self._goal_distance(timestep.state)

        def cond_fn(carry):
            _, _, goal_distance = carry
            return goal_distance < self.min_start_distance

        def body_fn(carry):
            rng, _, _ = carry
            rng, sample_key = jax.random.split(rng)
            sampled_timestep = self._env.reset(sample_key)
            goal_distance = self._goal_distance(sampled_timestep.state)
            return rng, sampled_timestep, goal_distance

        _, timestep, _ = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (key, timestep, distance),
        )
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
        def reset_branch(_):
            reset_obs, reset_timestep = self.reset(key, vision_radius=vision_radius)
            reward = jnp.asarray(0.0, dtype=jnp.float32)
            done = jnp.asarray(False)
            info = {
                "return": reset_timestep.info.get("return", reward),
                "task_reward": reward,
                "progress_reward": reward,
                "step_penalty": reward,
                "goal_distance": self._goal_distance(reset_timestep.state),
            }
            return reset_obs, reset_timestep, reward, done, info

        def step_branch(_):
            old_distance = self._goal_distance(timestep.state)
            next_timestep = self._env.step(timestep, action)
            new_distance = self._goal_distance(next_timestep.state)
            obs = self._split_observations(next_timestep, vision_radius)
            task_reward = next_timestep.reward.astype(jnp.float32)
            progress_reward = (old_distance - new_distance) * self.progress_reward_scale
            step_penalty = self.step_penalty
            reward = task_reward + progress_reward - step_penalty
            done = next_timestep.is_done()
            info = dict(next_timestep.info)
            info["task_reward"] = task_reward
            info["progress_reward"] = progress_reward
            info["step_penalty"] = step_penalty
            info["goal_distance"] = new_distance
            return obs, next_timestep, reward, done, info

        return jax.lax.cond(timestep.is_done(), reset_branch, step_branch, operand=None)

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
