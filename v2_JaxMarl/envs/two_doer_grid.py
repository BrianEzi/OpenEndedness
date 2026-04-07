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
    target_items: jnp.ndarray
    shuffled_menus: jnp.ndarray
    selected_correctly: jnp.ndarray
    has_selected: jnp.ndarray
    has_arrived: jnp.ndarray


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
        arrival_reward: float = 0.5,
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
        self.arrival_reward = jnp.asarray(arrival_reward, dtype=jnp.float32)
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
        self.item_bank = self._build_item_bank()

    @property
    def num_actions(self) -> int:
        # stay, up, right, down, left, pick_0, pick_1, pick_2, pick_3
        return 9
        
    def _build_item_bank(self) -> jnp.ndarray:
        colors = jnp.asarray([
            [1.0, 0.2, 0.2],
            [0.2, 1.0, 0.2],
            [0.2, 0.4, 1.0],
            [1.0, 1.0, 0.2],
        ], dtype=jnp.float32)
        shape0 = jnp.ones((5, 5), dtype=jnp.float32)
        shape1 = jnp.zeros((5, 5), dtype=jnp.float32)
        shape1 = shape1.at[2, 1:4].set(1.0)
        shape1 = shape1.at[1:4, 2].set(1.0)
        shape2 = jnp.zeros((5, 5), dtype=jnp.float32)
        shape2 = shape2.at[1, 1].set(1.0)
        shape2 = shape2.at[2, 2].set(1.0)
        shape2 = shape2.at[3, 3].set(1.0)
        shape2 = shape2.at[1, 3].set(1.0)
        shape2 = shape2.at[3, 1].set(1.0)
        shape3 = jnp.ones((5, 5), dtype=jnp.float32)
        shape3 = shape3.at[1:4, 1:4].set(0.0)
        shapes = jnp.stack([shape0, shape1, shape2, shape3])
        bank = jnp.zeros((16, 5, 5, 3), dtype=jnp.float32)
        for c in range(4):
            for s in range(4):
                bank = bank.at[c * 4 + s].set(shapes[s, :, :, None] * colors[c])
        return bank

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
        
        target_images = self.item_bank[state.target_items]
        menu_images = self.item_bank[state.shuffled_menus]
        menu_images = jnp.where(goals_reached[:, None, None, None, None], menu_images, 0.0)

        return {
            "global_map": global_map,
            "symbolic_state": symbolic_state,
            "local_views": local_views,
            "proprioceptions": proprioceptions,
            "target_images": target_images,
            "menu_images": menu_images,
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
        
        key_target_a, key_target_b, key_menu_a, key_menu_b = jax.random.split(key, 4)
        target_a = jax.random.randint(key_target_a, (), 0, 16)
        target_b = jax.random.randint(key_target_b, (), 0, 16)
        
        def make_menu(rng_key, target):
            p = jnp.where(jnp.arange(16) == target, 0.0, 1.0 / 15.0)
            distractors = jax.random.choice(rng_key, jnp.arange(16), shape=(3,), replace=False, p=p)
            menu = jnp.concatenate([jnp.array([target]), distractors])
            _, shuffle_key = jax.random.split(rng_key)
            return jax.random.permutation(shuffle_key, menu)
            
        menu_a = make_menu(key_menu_a, target_a)
        menu_b = make_menu(key_menu_b, target_b)
        
        state = TwoDoerState(
            positions=positions,
            goals=goals,
            step_count=jnp.asarray(0, dtype=jnp.int32),
            done=jnp.asarray(False),
            target_items=jnp.array([target_a, target_b]),
            shuffled_menus=jnp.stack([menu_a, menu_b]),
            selected_correctly=jnp.array([False, False]),
            has_selected=jnp.array([False, False]),
            has_arrived=jnp.array([False, False]),
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
                "failed": jnp.asarray(False),
            }
            return reset_obs, reset_state, zeros, jnp.asarray(False), info

        def step_branch(_):
            nav_actions = jnp.where(actions < 5, actions, 0)
            select_actions = actions - 5
            is_selecting = actions >= 5
            at_goal = jnp.all(state.positions == state.goals, axis=-1)

            valid_selection = jnp.logical_and(is_selecting, at_goal)
            valid_selection = jnp.logical_and(valid_selection, ~state.has_selected)

            chosen_item = jnp.take_along_axis(state.shuffled_menus, select_actions[:, None], axis=1)[:, 0]
            is_correct = chosen_item == state.target_items
            
            new_has_selected = jnp.logical_or(state.has_selected, valid_selection)
            new_selected_correctly = jnp.logical_or(state.selected_correctly, jnp.logical_and(valid_selection, is_correct))
            
            old_distances = self._manhattan_distance(state.positions, state.goals)
            next_positions, wall_hits, collision_blocks = self._resolve_actions(state.positions, nav_actions)
            next_positions = jnp.where(is_selecting[:, None], state.positions, next_positions)
            
            new_distances = self._manhattan_distance(next_positions, state.goals)
            progress_reward_per_doer = (old_distances - new_distances) * self.progress_reward_scale
            progress_reward = progress_reward_per_doer.sum()
            wall_penalty = self.wall_penalty * wall_hits.astype(jnp.float32).sum()
            collision_penalty = (
                self.collision_penalty * collision_blocks.astype(jnp.float32).sum()
            )
            
            arrived_this_step = jnp.logical_and(
                jnp.all(next_positions == state.goals, axis=-1),
                ~state.has_arrived
            )
            arrival_reward = (arrived_this_step.astype(jnp.float32) * self.arrival_reward).sum()
            new_has_arrived = jnp.logical_or(state.has_arrived, arrived_this_step)
            
            success = jnp.all(new_selected_correctly)
            failed = jnp.any(jnp.logical_and(new_has_selected, ~new_selected_correctly))
            
            task_reward = jnp.where(success, self.goal_reward, jnp.asarray(0.0, dtype=jnp.float32))
            reward = (
                task_reward
                + progress_reward
                + arrival_reward
                - self.step_penalty
                - wall_penalty
                - collision_penalty
            )
            next_state = TwoDoerState(
                positions=next_positions,
                goals=state.goals,
                step_count=state.step_count + 1,
                done=jnp.logical_or(jnp.logical_or(success, failed), state.step_count + 1 >= self.max_steps),
                target_items=state.target_items,
                shuffled_menus=state.shuffled_menus,
                selected_correctly=new_selected_correctly,
                has_selected=new_has_selected,
                has_arrived=new_has_arrived,
            )
            info = {
                "task_reward": task_reward,
                "progress_reward_per_doer": progress_reward_per_doer,
                "step_penalty": self.step_penalty,
                "wall_penalty": wall_penalty,
                "collision_penalty": collision_penalty,
                "goal_distance": new_distances,
                "success": success,
                "failed": failed,
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
            triangle_mask = right_triangle if agent_idx == 0 else left_triangle
            tile_view[triangle_mask] = np.asarray(self._agent_colors[agent_idx])
            frame[y0:(row + 1) * tile, x0:(col + 1) * tile] = tile_view

        # Render Item Targets at goals
        bank = np.asarray(self.item_bank)
        for agent_idx in range(self.num_doers):
            target_item = int(np.asarray(state.target_items)[agent_idx])
            target_img = bank[target_item] # (5, 5, 3)
            # scale 5x5 to inset region
            goal_row, goal_col = np.asarray(state.goals[agent_idx]).tolist()
            y0 = goal_row * tile
            x0 = goal_col * tile
            # upscale target_img to 15x15
            target_upscaled = np.repeat(np.repeat(target_img, 3, axis=0), 3, axis=1)
            offset_y, offset_x = (tile - 15) // 2, (tile - 15) // 2
            
            # Place target in center of goal
            frame[
                y0 + offset_y : y0 + offset_y + 15, 
                x0 + offset_x : x0 + offset_x + 15
            ] = target_upscaled
            
            # Render menu options when goal is reached
            if bool(np.asarray(state.positions[agent_idx] == state.goals[agent_idx]).all()):
                menus = np.asarray(state.shuffled_menus[agent_idx])
                for m_idx, option_id in enumerate(menus):
                    opt_img = bank[option_id]
                    opt_upscaled = np.repeat(np.repeat(opt_img, 3, axis=0), 3, axis=1)
                    # draw menu below the grid or adjacent
                    my0 = y0 - (m_idx + 1) * tile if agent_idx == 1 else y0 + (m_idx + 1) * tile
                    # clip boundary
                    my0 = max(0, min(my0, (self.grid_height - 1) * tile))
                    frame[my0 + offset_y : my0 + offset_y + 15, x0 + offset_x : x0 + offset_x + 15] = opt_upscaled

        return (frame * 255.0).astype(np.uint8)
