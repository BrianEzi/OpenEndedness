import jax
import jax.numpy as jnp
from flax import struct
import chex

@struct.dataclass
class EnvState:
    doer_pos: chex.Array  # (2,) int32
    target_pos: chex.Array  # (2,) int32
    last_message: chex.Array  # (5,) float32
    time: int

class BlindFetchEnv:
    def __init__(self):
        self.grid_size = 5
        self.max_steps = 50

    def reset(self, rng: chex.PRNGKey) -> EnvState:
        rng_doer, rng_target = jax.random.split(rng)
        # Random doer position
        doer_pos = jax.random.randint(rng_doer, (2,), 0, self.grid_size)
        # Fixed target position as per requirements (or random if desired, but sticking to "Fixed or random" -> let's make it fixed [4,4] for simplicity of phase 1 proof)
        # Requirement said: "Fixed or random (e.g., [4,4])". I'll stick to fixed [4,4] for Phase 1 stability.
        target_pos = jnp.array([4, 4], dtype=jnp.int32) 
        
        last_message = jnp.zeros((5,), dtype=jnp.float32)
        return EnvState(
            doer_pos=doer_pos,
            target_pos=target_pos,
            last_message=last_message,
            time=0
        )

    def step(self, state: EnvState, actions: dict, rng: chex.PRNGKey):
        # actions: {'seer_msg': (5,), 'doer_move': int}
        seer_msg = actions['seer_msg']
        doer_move = actions['doer_move']

        # Movements: 0=Stay, 1=Up, 2=Down, 3=Left, 4=Right
        # Grid: (0,0) is top-left usually in matrix, but let's assume standard cartesian or matrix indices.
        # Let's map: 1=Up (y-1), 2=Down (y+1), 3=Left (x-1), 4=Right (x+1)
        # Array is [x, y] usually? Or [row, col]?
        # Requirement says "2D grid world". Let's assume [x, y].
        # 1=Up -> y+1, 2=Down -> y-1, 3=Left -> x-1, 4=Right -> x+1? 
        # Or Matrix: 1=Up -> row-1, 2=Down -> row+1... 
        # Let's stick to standard "Grid World" conventions. 
        # Usually [x, y] where x is horizontal, y is vertical.
        # 1=Up (y+1), 2=Down (y-1), 3=Left (x-1), 4=Right (x+1)
        # Wait, if clipped to 0-4, and [4,4] is target.
        # Let's define: 0=Stay, 1=Up (y-1), 2=Down (y+1), 3=Left (x-1), 4=Right (x+1). (Matrix coordinates, top-left 0,0)
        
        moves = jnp.array([
            [0, 0],   # 0: Stay
            [0, -1],  # 1: Up
            [0, 1],   # 2: Down
            [-1, 0],  # 3: Left
            [1, 0]    # 4: Right
        ], dtype=jnp.int32)

        delta = moves[doer_move]
        new_pos = state.doer_pos + delta
        new_pos = jnp.clip(new_pos, 0, self.grid_size - 1)

        # Check target reached
        target_reached = jnp.array_equal(new_pos, state.target_pos)

        # Reward
        # +1.0 if doer_pos == target_pos (at the end of step), -0.01 per step
        reward = jax.lax.select(target_reached, 1.0, -0.01)
        # Note: If already done, we shouldn't get more reward, but the loop usually handles masking. 
        # The step function just computes immediate reward.
        
        # Done condition
        new_time = state.time + 1
        # Done if target reached OR max steps exceeded
        done = jnp.logical_or(target_reached, new_time >= self.max_steps)

        new_state = state.replace(
            doer_pos=new_pos,
            last_message=seer_msg,
            time=new_time
        )
        
        info = {'reached_target': target_reached}

        return new_state, reward, done, info

    def get_obs(self, state: EnvState):
        # Seer: [doer_pos, target_pos] normalized
        norm_doer = state.doer_pos.astype(jnp.float32) / (self.grid_size - 1)
        norm_target = state.target_pos.astype(jnp.float32) / (self.grid_size - 1)
        seer_obs = jnp.concatenate([norm_doer, norm_target], axis=-1)

        # Doer: last_message
        doer_obs = state.last_message

        return {
            'seer': seer_obs,
            'doer': doer_obs
        }
