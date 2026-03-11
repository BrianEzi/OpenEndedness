import jax
import jax.numpy as jnp
import optax
import wandb
import flax.linen as nn
from flax.training.train_state import TrainState

# Import our custom modules
from models.seer import Seer
from models.doer import Doer
from envs.wrappers import AsymmetricOvercookedWrapper
from training.loop import generate_trajectory
from agents.mappo import update_actor, update_critic, Transition

# For the sake of a complete script, here is a pragmatic, standard Critic
class GlobalCritic(nn.Module):
    """Evaluates the global state to guide learning (CTDE)."""
    @nn.compact
    def __call__(self, global_map: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(features=32, kernel_size=(3, 3))(global_map)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        value = nn.Dense(features=1)(x)
        return value

def main():
    # 1. Configuration and Logging
    config = {
        "learning_rate": 3e-4,
        "num_envs": 1, # Kept at 1 for initial debugging, scale up later
        "num_steps": 128,
        "total_timesteps": 1_000_000,
        "fsq_levels": [5, 5, 5], # Defines the categorical hypercube
        "seed": 42
    }
    
    wandb.init(project="emergent-comms-fsq", config=config)
    
    # 2. PRNG Key Initialization
    # JAX requires explicit, rigorous management of randomness
    rng = jax.random.PRNGKey(config["seed"])
    rng, seer_init_rng, doer_init_rng, critic_init_rng, env_rng = jax.random.split(rng, 5)

    # 3. Environment Instantiation
    # For a real implementation, you would import jaxmarl.make("overcooked") here
    class DummyEnv: # Placeholder for the jaxmarl environment
        def reset(self, key): return {}, None
        def step(self, key, state, action): return {}, None, 0.0, False, {}
        
    raw_env = DummyEnv()
    env = AsymmetricOvercookedWrapper(raw_env)

    # 4. Network Instantiation
    seer = Seer(fsq_levels=config["fsq_levels"])
    doer = Doer(fsq_levels=config["fsq_levels"], num_actions=env.num_actions)
    critic = GlobalCritic()

    # 5. Parameter Initialization (Dummy Forward Passes)
    # We must pass data of the correct shape to initialize the Flax parameters
    dummy_map = jnp.zeros((1, 15, 15, 3))
    dummy_sym = jnp.zeros((1, 2))
    dummy_local = jnp.zeros((1, 3, 3, 3))
    dummy_prop = jnp.zeros((1, 1))
    dummy_msg = jnp.zeros((1, len(config["fsq_levels"])))
    
    seer_carry = seer.initialize_carry(1, 128)
    doer_carry = doer.initialize_carry(1, 128)

    seer_params = seer.init(seer_init_rng, seer_carry, dummy_map, dummy_sym)["params"]
    doer_params = doer.init(doer_init_rng, doer_carry, dummy_local, dummy_prop, dummy_msg)["params"]
    critic_params = critic.init(critic_init_rng, dummy_map)["params"]

    # Group parameters for the execution loop
    params = {"seer": seer_params, "doer": doer_params, "critic": critic_params}

    # 6. Optimizer and TrainState Setup
    # Optax provides the gradient transformation tools
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=config["learning_rate"], eps=1e-5)
    )

    # Since the Seer and Doer act cooperatively to generate the trajectory,
    # we can conceptually treat them as a single "Actor" policy for the optimizer.
    # In a more advanced setup, you might give them separate optimizers.
    actor_state = TrainState.create(
        apply_fn=None, # We use specific apply fns in the update step
        params={"seer": seer_params, "doer": doer_params},
        tx=tx
    )
    
    critic_state = TrainState.create(
        apply_fn=critic.apply,
        params=critic_params,
        tx=tx
    )

    # 7. The Main Training Loop
    num_updates = config["total_timesteps"] // config["num_steps"]
    
    print("Starting training...")
    for update in range(num_updates):
        rng, rollout_rng = jax.random.split(rng)
        
        # A. Collect Trajectory
        # This compiles and runs the entire rollout in C++ / XLA
        final_runner_state, trajectory_batch = generate_trajectory(
            env, params, rollout_rng, config["num_steps"], 
            seer.apply, doer.apply, critic.apply
        )
        
        # B. Generalized Advantage Estimation (GAE) would be calculated here
        # For brevity, we assume trajectory_batch has 'advantage' and 'return_val' populated
        
        # C. Update Networks
        actor_state, actor_metrics = update_actor(actor_state, trajectory_batch)
        critic_state, critic_metrics = update_critic(critic_state, trajectory_batch)
        
        # Sync updated parameters back to the params dictionary for the next rollout
        params["seer"] = actor_state.params["seer"]
        params["doer"] = actor_state.params["doer"]
        params["critic"] = critic_state.params
        
        # D. Logging
        if update % 10 == 0:
            wandb.log({
                "update": update,
                "actor_loss": actor_metrics.get("actor_loss", 0.0),
                "entropy": actor_metrics.get("entropy", 0.0),
                "critic_loss": critic_metrics.get("critic_loss", 0.0),
                "mean_reward": trajectory_batch.reward.mean()
            })
            print(f"Update {update}/{num_updates} | Reward: {trajectory_batch.reward.mean():.3f}")

if __name__ == "__main__":
    main()