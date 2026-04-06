import jax
import jax.numpy as jnp
import numpy as np
from flax import struct


@struct.dataclass
class TwoDoerState:
    positions: jnp.ndarray
    goals: jnp.ndarray
    step_count: jnp.ndarray
    done: jnp.ndarray


class TwoDoerBottleneckEnv:
    """Two embodied Doers must swap sides through a single choke point."""

    def __init__(
        self,
        grid_size: int = 9,
        local_view_size: int = 3,
        max_steps: int = 48,
        progress_reward_scale: float = 0.05,
        goal_reward: float = 1.0,
        step_penalty: float = 0.03,
        wall_penalty: float = 0.02,
        collision_penalty: float = 0.05,
    ):
        if grid_size < 7 or grid_size % 2 == 0:
            raise ValueError("grid_size must be an odd integer >= 7.")
        if local_view_size % 2 == 0:
            raise ValueError("local_view_size must be odd.")

        self.grid_size = int(grid_size)
        self.local_view_size = int(local_view_size)
        self.max_steps = int(max_steps)
        self.progress_reward_scale = jnp.asarray(progress_reward_scale, dtype=jnp.float32)
        self.goal_reward = jnp.asarray(goal_reward, dtype=jnp.float32)
        self.step_penalty = jnp.asarray(step_penalty, dtype=jnp.float32)
        self.wall_penalty = jnp.asarray(wall_penalty, dtype=jnp.float32)
        self.collision_penalty = jnp.asarray(collision_penalty, dtype=jnp.float32)
        self.num_doers = 2
        self._view_radius = self.local_view_size // 2
        self._bottleneck_row = self.grid_size // 2
        self._bottleneck_col = self.grid_size // 2
        self._left_col = 1
        self._right_col = self.grid_size - 2
        self._candidate_rows = jnp.asarray([1, 2, self.grid_size - 3, self.grid_size - 2], dtype=jnp.int32)
        self._wall_map = self._build_wall_map()
        self._goal_colors = jnp.asarray(
            [
                [0.90, 0.25, 0.25],
                [0.20, 0.70, 0.30],
            ],
            dtype=jnp.float32,
        )
        self._agent_colors = jnp.asarray(
            [
                [0.85, 0.10, 0.10],
                [0.05, 0.45, 0.95],
            ],
            dtype=jnp.float32,
        )

    @property
    def num_actions(self) -> int:
        # stay, up, right, down, left
        return 5

    def _build_wall_map(self) -> jnp.ndarray:
        wall_map = jnp.zeros((self.grid_size, self.grid_size), dtype=bool)
        wall_map = wall_map.at[0, :].set(True)
        wall_map = wall_map.at[-1, :].set(True)
        wall_map = wall_map.at[:, 0].set(True)
        wall_map = wall_map.at[:, -1].set(True)
        wall_map = wall_map.at[:, self._bottleneck_col].set(True)
        wall_map = wall_map.at[self._bottleneck_row, self._bottleneck_col].set(False)
        return wall_map

    @staticmethod
    def _manhattan_distance(positions: jnp.ndarray, goals: jnp.ndarray) -> jnp.ndarray:
        return jnp.abs(positions - goals).sum(axis=-1).astype(jnp.float32)

    def _compose_global_map(self, state: TwoDoerState) -> jnp.ndarray:
        global_map = jnp.zeros((self.grid_size, self.grid_size, 5), dtype=jnp.float32)
        global_map = global_map.at[:, :, 0].set(self._wall_map.astype(jnp.float32))
        global_map = global_map.at[state.positions[0, 0], state.positions[0, 1], 1].set(1.0)
        global_map = global_map.at[state.positions[1, 0], state.positions[1, 1], 2].set(1.0)
        global_map = global_map.at[state.goals[0, 0], state.goals[0, 1], 3].set(1.0)
        global_map = global_map.at[state.goals[1, 0], state.goals[1, 1], 4].set(1.0)
        return global_map

    def _extract_local_views(self, global_map: jnp.ndarray, positions: jnp.ndarray) -> jnp.ndarray:
        padded_map = jnp.pad(
            global_map,
            (
                (self._view_radius, self._view_radius),
                (self._view_radius, self._view_radius),
                (0, 0),
            ),
        )

        def slice_agent_view(position):
            row = position[0]
            col = position[1]
            return jax.lax.dynamic_slice(
                padded_map,
                (row, col, 0),
                (self.local_view_size, self.local_view_size, global_map.shape[-1]),
            )

        return jax.vmap(slice_agent_view)(positions)

    def _split_observations(self, state: TwoDoerState):
        global_map = self._compose_global_map(state)
        local_views = self._extract_local_views(global_map, state.positions)
        goals_reached = jnp.all(state.positions == state.goals, axis=-1).astype(jnp.float32)
        agent_identity = jnp.eye(self.num_doers, dtype=jnp.float32)
        proprioceptions = jnp.concatenate(
            [agent_identity, goals_reached[:, None]],
            axis=-1,
        )
        symbolic_state = jnp.concatenate(
            [
                state.positions.astype(jnp.float32).reshape(-1) / float(self.grid_size - 1),
                state.goals.astype(jnp.float32).reshape(-1) / float(self.grid_size - 1),
                goals_reached,
                jnp.asarray(
                    [state.step_count.astype(jnp.float32) / float(self.max_steps)],
                    dtype=jnp.float32,
                ),
            ],
            axis=0,
        )
        return {
            "global_map": global_map,
            "symbolic_state": symbolic_state,
            "local_views": local_views,
            "proprioceptions": proprioceptions,
        }

    def reset(self, key: jnp.ndarray):
        row_key_a, row_key_b = jax.random.split(key)
        row_a = self._candidate_rows[jax.random.randint(row_key_a, (), 0, self._candidate_rows.shape[0])]
        row_b = self._candidate_rows[jax.random.randint(row_key_b, (), 0, self._candidate_rows.shape[0])]
        positions = jnp.asarray(
            [
                [row_a, self._left_col],
                [row_b, self._right_col],
            ],
            dtype=jnp.int32,
        )
        goals = jnp.asarray(
            [
                [row_a, self._right_col],
                [row_b, self._left_col],
            ],
            dtype=jnp.int32,
        )
        state = TwoDoerState(
            positions=positions,
            goals=goals,
            step_count=jnp.asarray(0, dtype=jnp.int32),
            done=jnp.asarray(False),
        )
        return self._split_observations(state), state

    def reset_batch(self, keys: jnp.ndarray):
        return jax.vmap(self.reset)(keys)

    def _resolve_actions(self, positions: jnp.ndarray, actions: jnp.ndarray):
        deltas = jnp.asarray(
            [
                [0, 0],
                [-1, 0],
                [0, 1],
                [1, 0],
                [0, -1],
            ],
            dtype=jnp.int32,
        )
        raw_targets = positions + deltas[actions]
        target_walls = self._wall_map[raw_targets[:, 0], raw_targets[:, 1]]
        wall_hits = jnp.logical_and(actions != 0, target_walls)
        proposed = jnp.where(target_walls[:, None], positions, raw_targets)

        same_target = jnp.all(proposed[0] == proposed[1])
        swap_positions = jnp.logical_and(
            jnp.all(proposed[0] == positions[1]),
            jnp.all(proposed[1] == positions[0]),
        )
        a_into_b = jnp.logical_and(
            jnp.all(proposed[0] == positions[1]),
            jnp.all(proposed[1] == positions[1]),
        )
        b_into_a = jnp.logical_and(
            jnp.all(proposed[1] == positions[0]),
            jnp.all(proposed[0] == positions[0]),
        )
        collision = jnp.logical_or(jnp.logical_or(same_target, swap_positions), jnp.logical_or(a_into_b, b_into_a))

        collision_blocks = jnp.asarray(
            [
                jnp.logical_or(same_target, jnp.logical_or(swap_positions, a_into_b)),
                jnp.logical_or(same_target, jnp.logical_or(swap_positions, b_into_a)),
            ],
            dtype=bool,
        )
        final_positions = jnp.where(collision_blocks[:, None], positions, proposed)
        return final_positions, wall_hits, collision_blocks

    def step(self, key: jnp.ndarray, state: TwoDoerState, actions: jnp.ndarray):
        def reset_branch(_):
            reset_obs, reset_state = self.reset(key)
            zeros = jnp.asarray(0.0, dtype=jnp.float32)
            info = {
                "task_reward": zeros,
                "progress_reward_per_doer": jnp.zeros((self.num_doers,), dtype=jnp.float32),
                "step_penalty": zeros,
                "wall_penalty": zeros,
                "collision_penalty": zeros,
                "goal_distance": self._manhattan_distance(reset_state.positions, reset_state.goals),
                "success": jnp.asarray(False),
            }
            return reset_obs, reset_state, zeros, jnp.asarray(False), info

        def step_branch(_):
            old_distances = self._manhattan_distance(state.positions, state.goals)
            next_positions, wall_hits, collision_blocks = self._resolve_actions(state.positions, actions)
            new_distances = self._manhattan_distance(next_positions, state.goals)
            progress_reward_per_doer = (old_distances - new_distances) * self.progress_reward_scale
            progress_reward = progress_reward_per_doer.sum()
            wall_penalty = self.wall_penalty * wall_hits.astype(jnp.float32).sum()
            collision_penalty = (
                self.collision_penalty * collision_blocks.astype(jnp.float32).sum()
            )
            success = jnp.all(next_positions == state.goals)
            task_reward = jnp.where(success, self.goal_reward, jnp.asarray(0.0, dtype=jnp.float32))
            reward = (
                task_reward
                + progress_reward
                - self.step_penalty
                - wall_penalty
                - collision_penalty
            )
            next_state = TwoDoerState(
                positions=next_positions,
                goals=state.goals,
                step_count=state.step_count + 1,
                done=jnp.logical_or(success, state.step_count + 1 >= self.max_steps),
            )
            info = {
                "task_reward": task_reward,
                "progress_reward_per_doer": progress_reward_per_doer,
                "step_penalty": self.step_penalty,
                "wall_penalty": wall_penalty,
                "collision_penalty": collision_penalty,
                "goal_distance": new_distances,
                "success": success,
            }
            return self._split_observations(next_state), next_state, reward, next_state.done, info

        return jax.lax.cond(state.done, reset_branch, step_branch, operand=None)

    def step_batch(self, keys: jnp.ndarray, states: TwoDoerState, actions: jnp.ndarray):
        return jax.vmap(self.step)(keys, states, actions)

    def render(self, state: TwoDoerState) -> np.ndarray:
        frame = np.ones((self.grid_size, self.grid_size, 3), dtype=np.float32) * 0.96
        frame[np.asarray(self._wall_map)] = np.array([0.18, 0.18, 0.22], dtype=np.float32)

        for agent_idx in range(self.num_doers):
            goal_row, goal_col = np.asarray(state.goals[agent_idx]).tolist()
            frame[goal_row, goal_col] = np.asarray(self._goal_colors[agent_idx])

        for agent_idx in range(self.num_doers):
            row, col = np.asarray(state.positions[agent_idx]).tolist()
            frame[row, col] = np.asarray(self._agent_colors[agent_idx])

        frame[self._bottleneck_row, self._bottleneck_col] = np.array([0.98, 0.88, 0.45], dtype=np.float32)
        return (frame * 255.0).astype(np.uint8)
