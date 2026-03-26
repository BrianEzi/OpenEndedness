import jax
import jax.numpy as jnp
import numpy as np
from navix import observations


class NavixGridWrapper:
    """Expose Navix single-agent environments with the Seer/Doer observation split."""

    SEER_NAV_PHASE = 0
    COMMUNICATION_PHASE = 1

    def __init__(
        self,
        env,
        progress_reward_scale: float = 0.1,
        min_start_distance: float = 0.0,
        step_penalty: float = 0.0,
        bump_penalty: float = 0.1,
        doer_perception_level: int = 0,
    ):
        self._env = env
        self.progress_reward_scale = progress_reward_scale
        self.min_start_distance = jnp.asarray(min_start_distance, dtype=jnp.float32)
        self.step_penalty = jnp.asarray(step_penalty, dtype=jnp.float32)
        self.bump_penalty = jnp.asarray(bump_penalty, dtype=jnp.float32)
        self.doer_perception_level = int(doer_perception_level)

    @property
    def num_actions(self) -> int:
        # Navigation-only action space: turn left, turn right, move forward.
        return 3

    def _split_observations(
        self,
        timestep,
        vision_radius: jnp.ndarray,
        control_mode: jnp.ndarray,
    ):
        state = timestep.state
        global_map = observations.symbolic(state).astype(jnp.float32)
        full_local_view = observations.symbolic_first_person(state).astype(jnp.float32)

        player = state.get_player()
        goal = state.get_goals()
        center_row = full_local_view.shape[0] // 2
        center_col = full_local_view.shape[1] // 2
        local_view_3x3 = jax.lax.dynamic_slice(
            full_local_view,
            (center_row - 1, center_col - 1, 0),
            (3, 3, full_local_view.shape[-1]),
        )

        symbolic_state = jnp.array(
            [
                player.position[0],
                player.position[1],
                player.direction,
                player.pocket,
                goal.position[0, 0],
                goal.position[0, 1],
                (control_mode == self.SEER_NAV_PHASE).astype(jnp.float32),
            ],
            dtype=jnp.float32,
        )

        proprioception_full = jnp.array(
            [
                player.position[0],
                player.position[1],
                player.direction,
                player.pocket,
            ],
            dtype=jnp.float32,
        )

        if self.doer_perception_level == 0:
            local_view = local_view_3x3
            proprioception = proprioception_full
        elif self.doer_perception_level == 1:
            local_view = jnp.zeros_like(local_view_3x3)
            local_view = local_view.at[0, 1].set(local_view_3x3[0, 1])
            proprioception = proprioception_full
        elif self.doer_perception_level == 2:
            local_view = jnp.zeros_like(local_view_3x3)
            proprioception = jnp.array(
                [
                    player.position[0],
                    player.position[1],
                    0.0,
                    0.0,
                ],
                dtype=jnp.float32,
            )
        elif self.doer_perception_level == 3:
            local_view = jnp.zeros_like(local_view_3x3)
            proprioception = jnp.zeros_like(proprioception_full)
        else:
            raise ValueError(
                f"Unsupported doer_perception_level={self.doer_perception_level}. "
                "Use 0, 1, 2, or 3."
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

    def reset(
        self,
        key: jnp.ndarray,
        vision_radius: jnp.ndarray = jnp.array(3.0),
        control_mode: jnp.ndarray = jnp.array(COMMUNICATION_PHASE),
    ):
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
        obs = self._split_observations(timestep, vision_radius, control_mode)
        return obs, timestep

    def reset_batch(
        self,
        keys: jnp.ndarray,
        vision_radius: jnp.ndarray = jnp.array(3.0),
        control_mode: jnp.ndarray = jnp.array(COMMUNICATION_PHASE),
    ):
        return jax.vmap(self.reset, in_axes=(0, None, None))(keys, vision_radius, control_mode)

    def step(
        self,
        key: jnp.ndarray,
        timestep,
        action: jnp.ndarray,
        vision_radius: jnp.ndarray = jnp.array(3.0),
        control_mode: jnp.ndarray = jnp.array(COMMUNICATION_PHASE),
    ):
        def reset_branch(_):
            reset_obs, reset_timestep = self.reset(
                key,
                vision_radius=vision_radius,
                control_mode=control_mode,
            )
            reward = jnp.asarray(0.0, dtype=jnp.float32)
            done = jnp.asarray(False)
            info = {
                "return": reset_timestep.info.get("return", reward),
                "task_reward": reward,
                "progress_reward": reward,
                "step_penalty": reward,
                "bump_penalty": reward,
                "goal_distance": self._goal_distance(reset_timestep.state),
            }
            return reset_obs, reset_timestep, reward, done, info

        def step_branch(_):
            old_distance = self._goal_distance(timestep.state)
            old_position = timestep.state.get_player().position
            next_timestep = self._env.step(timestep, action)
            new_distance = self._goal_distance(next_timestep.state)
            new_position = next_timestep.state.get_player().position
            obs = self._split_observations(next_timestep, vision_radius, control_mode)
            task_reward = next_timestep.reward.astype(jnp.float32)
            progress_reward = (old_distance - new_distance) * self.progress_reward_scale
            step_penalty = self.step_penalty
            bumped = jnp.logical_and(
                action == 2,
                jnp.all(new_position == old_position),
            )
            bump_penalty = jnp.where(
                bumped,
                self.bump_penalty,
                jnp.asarray(0.0, dtype=jnp.float32),
            )
            reward = task_reward + progress_reward - step_penalty - bump_penalty
            done = next_timestep.is_done()
            info = dict(next_timestep.info)
            info["task_reward"] = task_reward
            info["progress_reward"] = progress_reward
            info["step_penalty"] = step_penalty
            info["bump_penalty"] = bump_penalty
            info["goal_distance"] = new_distance
            return obs, next_timestep, reward, done, info

        return jax.lax.cond(timestep.is_done(), reset_branch, step_branch, operand=None)

    def step_batch(
        self,
        keys: jnp.ndarray,
        timesteps,
        actions: jnp.ndarray,
        vision_radius: jnp.ndarray = jnp.array(3.0),
        control_mode: jnp.ndarray = jnp.array(COMMUNICATION_PHASE),
    ):
        return jax.vmap(self.step, in_axes=(0, 0, 0, None, None))(
            keys,
            timesteps,
            actions,
            vision_radius,
            control_mode,
        )

    def render(
        self,
        timestep,
        control_mode: int = COMMUNICATION_PHASE,
    ):
        frame = np.asarray(observations.rgb(timestep.state)).copy()
        grid = np.asarray(observations.symbolic(timestep.state))
        player = np.asarray(timestep.state.get_player().position).astype(np.int32)
        tile_h = max(frame.shape[0] // grid.shape[0], 1)
        tile_w = max(frame.shape[1] // grid.shape[1], 1)
        row, col = int(player[0]), int(player[1])
        y0 = row * tile_h + tile_h // 4
        y1 = min((row + 1) * tile_h - tile_h // 4, frame.shape[0])
        x0 = col * tile_w + tile_w // 4
        x1 = min((col + 1) * tile_w - tile_w // 4, frame.shape[1])
        color = (
            np.array([32, 96, 224], dtype=np.uint8)
            if control_mode == self.SEER_NAV_PHASE
            else np.array([0, 0, 0], dtype=np.uint8)
        )
        frame[y0:y1, x0:x1] = color
        return frame
