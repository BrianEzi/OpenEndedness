import jax
import jax.numpy as jnp
from PIL import Image
import numpy as np

def visualize_episode(env, params, rng, seer_apply_fn, doer_apply_fn, filename="episode.gif"):
    """Runs one episode and saves the rendering to a GIF."""
    frames = []
    
    # Initialize env and state
    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng)
    
    # Initialize LSTM carries 
    seer_carry = seer_apply_fn.initialize_carry(batch_size=1, hidden_size=128)
    doer_carry = doer_apply_fn.initialize_carry(batch_size=1, hidden_size=128)
    
    done = False
    step_count = 0
    
    while not done and step_count < 200:
        # Render the current state
        # jaxmarl render usually returns a numpy array (RGB)
        frame = env.render(state)
        frames.append(Image.fromarray(frame))
        
        # 1. Seer generates message based on Global Map [cite: 80, 132]
        seer_carry, message, _ = seer_apply_fn(
            {"params": params["seer"]}, 
            seer_carry, 
            obs["global_map"], 
            obs["symbolic_state"]
        )
        
        # 2. Doer takes action based on message and local/blind view [cite: 82, 138]
        doer_carry, action_logits = doer_apply_fn(
            {"params": params["doer"]}, 
            doer_carry, 
            obs["local_view"], 
            obs["proprioception"], 
            message
        )
        
        # Select action (Greedy for visualization)
        action = jnp.argmax(action_logits, axis=-1)
        
        # 3. Step Environment
        rng, step_rng = jax.random.split(rng)
        obs, state, reward, done, _ = env.step(step_rng, state, action)
        
        # Print logs to see communication in real-time
        print(f"Step {step_count} | Message: {message} | Action: {action} | Reward: {reward}")
        step_count += 1

    # Save animation
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=100, loop=0)
    print(f"Episode saved to {filename}")