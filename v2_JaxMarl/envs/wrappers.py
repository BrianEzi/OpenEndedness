import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any

class AsymmetricOvercookedWrapper:
    """
    Wraps a jaxmarl Overcooked environment to enforce the Seer-Doer split.
    Guarantees strict sensory deprivation for the Doer agent.
    """
    def __init__(self, env):
        self._env = env
        # In standard jaxmarl Overcooked, agents are usually 'agent_0' and 'agent_1'
        self.seer_id = 'agent_0'
        self.doer_id = 'agent_1'

    @property
    def num_actions(self) -> int:
        return self._env.action_space(self.doer_id).n

    def _split_observations(self, state: Any, raw_obs: Dict[str, jnp.ndarray], vision_radius: jnp.ndarray) -> Dict[str, Any]:
        """
        Transforms symmetric jaxmarl observations into structurally asymmetric inputs.
        Supports vision curriculum via vision_radius.
        """
        # 1. Seer (Prefrontal Cortex) Data Extraction
        # The Seer requires the Global Map View and Symbolic States[cite: 131, 132].
        # Depending on the specific jaxmarl layout, the raw observation might already 
        # be the global grid, or we might need to extract it from the environment state.
        global_map = state.maze_map # Example extraction of the underlying grid
        
        # Placeholder for symbolic state (e.g., recipe requirements, time remaining)
        # Using time and current agent inventory as symbolic state
        symbolic_state = jnp.array([state.time, state.agent_inv[0], state.agent_inv[1]]) 

        # 2. Doer (Motor Cortex) Data Extraction
        # The Doer requires an Egocentric 3x3 grid (or total blindness) and proprioception[cite: 137, 138].
        doer_idx = 1 if self.doer_id == 'agent_1' else 0
        doer_pos = state.agent_pos[doer_idx]
        doer_dir = state.agent_dir[doer_idx]
        
        local_view = raw_obs[self.doer_id]
        
        # Local view masking based on vision_radius
        # raw_obs is map shape (H, W, C).
        H, W = local_view.shape[0], local_view.shape[1]
        yy, xx = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij')
        
        # In jaxmarl, agent_pos is [x, y]. yy is height index (y), xx is width index (x).
        doer_x, doer_y = state.agent_pos[doer_idx][0], state.agent_pos[doer_idx][1]
        dist = jnp.maximum(jnp.abs(xx - doer_x), jnp.abs(yy - doer_y)) # Chebyshev distance
        mask = (dist <= vision_radius)[..., None]
        local_view = jnp.where(mask, local_view, 0)

        # Proprioception: e.g., Is the Doer holding an onion or a plate? [cite: 138]
        proprioception = jnp.array([state.agent_inv[doer_idx]])

        # 3. Restructure for our custom rollout loop
        return {
            "global_map": global_map,
            "symbolic_state": symbolic_state,
            "local_view": local_view,
            "proprioception": proprioception
        }

    def reset(self, key: jnp.ndarray, vision_radius: jnp.ndarray = jnp.array(2.0)) -> Tuple[Dict[str, Any], Any]:
        """Resets the environment and returns the asymmetric observation dictionary."""
        raw_obs, state = self._env.reset(key)
        asymmetric_obs = self._split_observations(state, raw_obs, vision_radius)
        return asymmetric_obs, state

    def step(
        self, 
        key: jnp.ndarray, 
        state: Any, 
        doer_action: jnp.ndarray,
        vision_radius: jnp.ndarray = jnp.array(2.0)
    ) -> Tuple[Dict[str, Any], Any, jnp.ndarray, jnp.ndarray, Dict]:
        """
        Steps the environment forward. 
        Notice that only the Doer submits a physical action.
        """
        # The Seer has zero physical agency[cite: 80]. 
        # Its "action" is the message passed directly to the Doer's neural network 
        # during the rollout loop, not to the environment simulator.
        env_actions = {
            self.seer_id: jnp.array(4), # 4 is STAY action in jaxmarl overcooked
            self.doer_id: doer_action
        }
        
        raw_obs, next_state, rewards, dones, info = self._env.step(key, state, env_actions)
        
        asymmetric_obs = self._split_observations(next_state, raw_obs, vision_radius)
        
        # Immediate Reward Shaping: Sub-Goal for picking up or dropping an item
        # Helps the Doer associate the Seer's message with object interaction
        doer_idx = 1 if self.doer_id == 'agent_1' else 0
        old_inv = state.agent_inv[doer_idx]
        new_inv = next_state.agent_inv[doer_idx]
        
        # Reward +0.1 for picking up an object
        picked_up = jnp.logical_and(old_inv == 0, new_inv != 0)
        pickup_reward = jnp.where(picked_up, 0.1, 0.0)
        
        # Reward +0.1 for dropping an object (e.g. in pot or counter)
        dropped = jnp.logical_and(old_inv != 0, new_inv == 0)
        drop_reward = jnp.where(dropped, 0.1, 0.0)
        
        # In a fully cooperative game like Overcooked, rewards are usually shared
        shared_reward = rewards[self.doer_id] + pickup_reward + drop_reward
        shared_done = dones['__all__']
        
        return asymmetric_obs, next_state, shared_reward, shared_done, info