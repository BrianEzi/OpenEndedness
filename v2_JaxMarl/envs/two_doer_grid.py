import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

UNSET_TWO_DOER_POSITIONS = jnp.full((2, 2), -1, dtype=jnp.int32)


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
        grid_height: int = 10,
        grid_width: int = 12,
        local_view_size: int = 3,
        corridor_length: int = 3,
        max_steps: int = 48,
        progress_reward_scale: float = 0.05,
        goal_reward: float = 1.0,
        step_penalty: float = 0.03,
        wall_penalty: float = 0.02,
        collision_penalty: float = 0.05,
        doer_perception_level: int = 2,
        render_tile_size: int = 24,
    ):
        if grid_height < 8:
            raise ValueError("grid_height must be >= 8.")
        if grid_width < 10:
            raise ValueError("grid_width must be >= 10.")
        if local_view_size % 2 == 0:
            raise ValueError("local_view_size must be odd.")
        if corridor_length < 1:
            raise ValueError("corridor_length must be >= 1.")
        if grid_width - corridor_length - 2 < 4:
            raise ValueError("grid_width is too small for the requested corridor_length.")

        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.local_view_size = int(local_view_size)
        self.corridor_length = int(corridor_length)
        self.max_steps = int(max_steps)
        self.progress_reward_scale = jnp.asarray(progress_reward_scale, dtype=jnp.float32)
        self.goal_reward = jnp.asarray(goal_reward, dtype=jnp.float32)
        self.step_penalty = jnp.asarray(step_penalty, dtype=jnp.float32)
        self.wall_penalty = jnp.asarray(wall_penalty, dtype=jnp.float32)
        self.collision_penalty = jnp.asarray(collision_penalty, dtype=jnp.float32)
        self.doer_perception_level = int(doer_perception_level)
        self.render_tile_size = int(render_tile_size)
        self.num_doers = 2
        self._view_radius = self.local_view_size // 2
        self._inner_width = self.grid_width - 2
        self._room_width = (self._inner_width - self.corridor_length) // 2
        self._extra_width = self._inner_width - self.corridor_length - 2 * self._room_width
        self._left_room_start_col = 1
        self._left_room_end_col = self._left_room_start_col + self._room_width
        self._corridor_row = self.grid_height // 2
        self._corridor_start_col = self._left_room_end_col
        self._corridor_end_col = self._corridor_start_col + self.corridor_length
        self._right_room_start_col = self._corridor_end_col
        self._right_room_end_col = self._right_room_start_col + self._room_width
        self._extra_wall_start_col = self._right_room_end_col
        self._extra_wall_end_col = self.grid_width - 1
        self._left_col = 1
        self._right_col = self._right_room_end_col - 1
        self._candidate_rows = jnp.asarray([1, 2, self.grid_height - 3, self.grid_height - 2], dtype=jnp.int32)
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
        wall_map = jnp.zeros((self.grid_height, self.grid_width), dtype=bool)
        wall_map = wall_map.at[0, :].set(True)
        wall_map = wall_map.at[-1, :].set(True)
        wall_map = wall_map.at[:, 0].set(True)
        wall_map = wall_map.at[:, -1].set(True)
        corridor_cols = jnp.arange(self._corridor_start_col, self._corridor_end_col)
        wall_map = wall_map.at[:, corridor_cols].set(True)
        wall_map = wall_map.at[self._corridor_row, corridor_cols].set(False)
        if self._extra_width > 0:
            extra_cols = jnp.arange(self._extra_wall_start_col, self._extra_wall_end_col)
            wall_map = wall_map.at[:, extra_cols].set(True)
        return wall_map

    @staticmethod
    def _manhattan_distance(positions: jnp.ndarray, goals: jnp.ndarray) -> jnp.ndarray:
        return jnp.abs(positions - goals).sum(axis=-1).astype(jnp.float32)

    def _compose_global_map(self, state: TwoDoerState) -> jnp.ndarray:
        global_map = jnp.zeros((self.grid_height, self.grid_width, 5), dtype=jnp.float32)
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
        full_local_views = self._extract_local_views(global_map, state.positions)
        goals_reached = jnp.all(state.positions == state.goals, axis=-1).astype(jnp.float32)
        agent_identity = jnp.eye(self.num_doers, dtype=jnp.float32)
        position_features = jnp.stack(
            [
                state.positions[:, 0].astype(jnp.float32) / float(self.grid_height - 1),
                state.positions[:, 1].astype(jnp.float32) / float(self.grid_width - 1),
            ],
            axis=-1,
        )
        proprioception_full = jnp.concatenate(
            [position_features, agent_identity, goals_reached[:, None]],
            axis=-1,
        )
        if self.doer_perception_level == 2:
            local_views = jnp.zeros_like(full_local_views)
            proprioceptions = proprioception_full
        elif self.doer_perception_level == 3:
            local_views = jnp.zeros_like(full_local_views)
            proprioceptions = jnp.concatenate(
                [
                    jnp.zeros_like(position_features),
                    agent_identity,
                    jnp.zeros_like(goals_reached[:, None]),
                ],
                axis=-1,
            )
        else:
            raise ValueError(
                f"Unsupported doer_perception_level={self.doer_perception_level}. "
                "Use 2 or 3."
            )
        symbolic_state = jnp.concatenate(
            [
                jnp.asarray(
                    [
                        state.positions[0, 0] / float(self.grid_height - 1),
                        state.positions[0, 1] / float(self.grid_width - 1),
                        state.positions[1, 0] / float(self.grid_height - 1),
                        state.positions[1, 1] / float(self.grid_width - 1),
                    ],
                    dtype=jnp.float32,
                ),
                jnp.asarray(
                    [
                        state.goals[0, 0] / float(self.grid_height - 1),
                        state.goals[0, 1] / float(self.grid_width - 1),
                        state.goals[1, 0] / float(self.grid_height - 1),
                        state.goals[1, 1] / float(self.grid_width - 1),
                    ],
                    dtype=jnp.float32,
                ),
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

    def _goals_from_positions(self, positions: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(
            [
                [positions[0, 0], self._right_col],
                [positions[1, 0], self._left_col],
            ],
            dtype=jnp.int32,
        )

    def reset(
        self,
        key: jnp.ndarray,
        fixed_positions: jnp.ndarray = UNSET_TWO_DOER_POSITIONS,
    ):
        row_key_a, row_key_b = jax.random.split(key)
        row_a = self._candidate_rows[jax.random.randint(row_key_a, (), 0, self._candidate_rows.shape[0])]
        row_b = self._candidate_rows[jax.random.randint(row_key_b, (), 0, self._candidate_rows.shape[0])]
        sampled_positions = jnp.asarray(
            [
                [row_a, self._left_col],
                [row_b, self._right_col],
            ],
            dtype=jnp.int32,
        )
        positions = jnp.where(
            jnp.all(fixed_positions >= 0),
            fixed_positions.astype(jnp.int32),
            sampled_positions,
        )
        goals = self._goals_from_positions(positions)
        state = TwoDoerState(
            positions=positions,
            goals=goals,
            step_count=jnp.asarray(0, dtype=jnp.int32),
            done=jnp.asarray(False),
        )
        return self._split_observations(state), state

    def reset_batch(
        self,
        keys: jnp.ndarray,
        fixed_positions: jnp.ndarray = UNSET_TWO_DOER_POSITIONS,
    ):
        return jax.vmap(self.reset, in_axes=(0, None))(keys, fixed_positions)

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

    def step(
        self,
        key: jnp.ndarray,
        state: TwoDoerState,
        actions: jnp.ndarray,
        fixed_positions: jnp.ndarray = UNSET_TWO_DOER_POSITIONS,
    ):
        def reset_branch(_):
            reset_obs, reset_state = self.reset(key, fixed_positions=fixed_positions)
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

    def step_batch(
        self,
        keys: jnp.ndarray,
        states: TwoDoerState,
        actions: jnp.ndarray,
        fixed_positions: jnp.ndarray = UNSET_TWO_DOER_POSITIONS,
    ):
        return jax.vmap(self.step, in_axes=(0, 0, 0, None))(keys, states, actions, fixed_positions)

    def render(self, state: TwoDoerState) -> np.ndarray:
        tile = self.render_tile_size
        frame = np.ones((self.grid_height * tile, self.grid_width * tile, 3), dtype=np.float32) * 0.96

        wall_color = np.array([0.18, 0.18, 0.22], dtype=np.float32)
        corridor_color = np.array([0.98, 0.88, 0.45], dtype=np.float32)

        wall_map = np.asarray(self._wall_map)
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                y0 = row * tile
                y1 = (row + 1) * tile
                x0 = col * tile
                x1 = (col + 1) * tile
                if wall_map[row, col]:
                    frame[y0:y1, x0:x1] = wall_color

        for col in range(self._corridor_start_col, self._corridor_end_col):
            y0 = self._corridor_row * tile
            y1 = (self._corridor_row + 1) * tile
            x0 = col * tile
            x1 = (col + 1) * tile
            frame[y0:y1, x0:x1] = corridor_color

        for agent_idx in range(self.num_doers):
            goal_row, goal_col = np.asarray(state.goals[agent_idx]).tolist()
            y0 = goal_row * tile
            x0 = goal_col * tile
            inset = max(tile // 4, 2)
            frame[
                y0 + inset:(goal_row + 1) * tile - inset,
                x0 + inset:(goal_col + 1) * tile - inset,
            ] = np.asarray(self._agent_colors[agent_idx])

        yy, xx = np.mgrid[0:tile, 0:tile]
        left_triangle = np.logical_and(xx >= tile // 5, np.abs(yy - tile // 2) <= xx - tile // 5)
        right_triangle = np.logical_and(
            xx <= tile - tile // 5 - 1,
            np.abs(yy - tile // 2) <= (tile - tile // 5 - 1) - xx,
        )

        for agent_idx in range(self.num_doers):
            row, col = np.asarray(state.positions[agent_idx]).tolist()
            y0 = row * tile
            x0 = col * tile
            tile_view = frame[y0:(row + 1) * tile, x0:(col + 1) * tile]
            triangle_mask = left_triangle if agent_idx == 0 else right_triangle
            tile_view[triangle_mask] = np.asarray(self._agent_colors[agent_idx])
            frame[y0:(row + 1) * tile, x0:(col + 1) * tile] = tile_view

        return (frame * 255.0).astype(np.uint8)
