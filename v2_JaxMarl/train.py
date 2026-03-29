import jax
import jax.numpy as jnp
import optax
from pathlib import Path
import flax.linen as nn
from flax.training.train_state import TrainState

# Import our custom modules
from models.seer import Seer
from models.doer import Doer
from envs.navix_wrapper import NavixGridWrapper
from training.loop import generate_trajectory_and_gae, make_rollout_step
from agents.mappo import update_actor, update_critic, Transition
import navix as nx
from eval.metrics import compute_cic
from eval.visualize import visualize_episode
import numpy as np
import wandb
import pickle

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
        value = nn.Dense(features=2)(x)
        return value

def main():
    # 1. Configuration and Logging
    config = {
        "seer_learning_rate": 2e-4,
        "doer_learning_rate": 3e-4,
        "num_envs": 16,
        "num_steps": 128,
        "total_timesteps": 10_000_000,
        "env_id": "Navix-Empty-Random-8x8-v0",
        "fsq_levels": [4], # Defines the categorical hypercube
        "seed": 42,
        "follow_reward_scale": 0.3,
        "progress_reward_scale": 0.2,
        "cic_coef": 0.0,
        "doer_perception_level": 0,
        "curriculum_success_streak": 3,
        "min_start_distance": 1.0,
        "step_penalty": 0.01,
        "bump_penalty": 0.1,
        "visualize_every": 400,
        "visualize_max_steps": 30,
        "visualize_dir": "artifacts/episodes",
        "seer_solo_checkpoint": "artifacts/checkpoints/seer_solo_final.pkl",
        "msg_diversity_scale": 0.1,
    }
    
    wandb.init(entity="eleftheriaklk-ucl", project="brian_test", config=config)
    
    # 2. PRNG Key Initialization
    # JAX requires explicit, rigorous management of randomness
    rng = jax.random.PRNGKey(config["seed"])
    rng, seer_init_rng, doer_init_rng, critic_init_rng, env_rng = jax.random.split(rng, 5)

    # 3. Environment Instantiation
    raw_env = nx.make(config["env_id"])
    env = NavixGridWrapper(
        raw_env,
        progress_reward_scale=config["progress_reward_scale"],
        min_start_distance=config["min_start_distance"],
        step_penalty=config["step_penalty"],
        bump_penalty=config["bump_penalty"],
        doer_perception_level=config["doer_perception_level"],
    )

    # 4. Initial Environment Reset
    rng, env_rng = jax.random.split(rng, 2)
    reset_keys = jax.random.split(env_rng, config["num_envs"])
    # Give initial full vision radius of 3.0 for the 7x7 local crop
    env_obs, env_state = env.reset_batch(reset_keys, vision_radius=jnp.array(3.0))

    # 5. Network Instantiation
    seer = Seer(fsq_levels=config["fsq_levels"])
    doer = Doer(fsq_levels=config["fsq_levels"], num_actions=env.num_actions)
    critic = GlobalCritic()

    # 6. Parameter Initialization (Dummy Forward Passes)
    # We must pass data of the correct shape to initialize the Flax parameters
    dummy_map = env_obs["global_map"][:1]
    dummy_sym = env_obs["symbolic_state"][:1]
    dummy_local = env_obs["local_view"][:1]
    dummy_prop = env_obs["proprioception"][:1]
    dummy_msg = jnp.zeros((1, len(config["fsq_levels"])))
    
    init_seer_carry = seer.initialize_carry(1, 128)
    init_doer_carry = doer.initialize_carry(1, 128)

    seer_params = seer.init(seer_init_rng, init_seer_carry, dummy_map, dummy_sym)["params"]
    doer_params = doer.init(doer_init_rng, init_doer_carry, dummy_local, dummy_prop, dummy_msg)["params"]
    critic_params = critic.init(critic_init_rng, dummy_map)["params"]

    _ckpt_path = Path(config["seer_solo_checkpoint"])
    if _ckpt_path.exists():
        with open(_ckpt_path, "rb") as f:
            solo_params = pickle.load(f)
        keys_to_transfer = ["Conv_0", "Conv_1", "Dense_0", "LSTMCell_0"]
        seer_params = dict(seer_params)
        for k in keys_to_transfer:
            seer_params[k] = solo_params[k]
        print(f"Transferred SeerSolo weights: {keys_to_transfer}")
    else:
        print("Warning: SeerSolo checkpoint not found, using random init.")


    seer_carry = seer.initialize_carry(config["num_envs"], 128)
    doer_carry = doer.initialize_carry(config["num_envs"], 128)

    # Group parameters for the execution loop
    params = {"seer": seer_params, "doer": doer_params, "critic": critic_params}

    # 6. Optimizer and TrainState Setup
    # Separate learning rates for Seer and Doer to prevent Seer gradient explosion
    tx = optax.multi_transform(
        {
            'seer': optax.chain(
                optax.clip_by_global_norm(0.5),
                optax.adam(learning_rate=config["seer_learning_rate"], eps=1e-5)
            ),
            'doer': optax.chain(
                optax.clip_by_global_norm(0.5),
                optax.adam(learning_rate=config["doer_learning_rate"], eps=1e-5)
            ),
        },
        param_labels=lambda params: {k: k for k in params}
    )

    actor_state = TrainState.create(
        apply_fn=None,
        params={"seer": seer_params, "doer": doer_params},
        tx=tx
    )
    
    tx_critic = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=config["doer_learning_rate"], eps=1e-5)
    )
    critic_state = TrainState.create(
        apply_fn=critic.apply,
        params=critic_params,
        tx=tx_critic
    )

    step_fn = make_rollout_step(
        env,
        seer.apply,
        doer.apply,
        critic.apply,
        follow_reward_scale=config["follow_reward_scale"],
        msg_diversity_scale=config["msg_diversity_scale"],
    )

    success_streak = 0

    # 8. The Main Training Loop
    num_updates = config["total_timesteps"] // (config["num_steps"] * config["num_envs"])
    
    print(
        "Starting training... "
        f"(doer_perception_level={config['doer_perception_level']})"
    )
    for update in range(num_updates):
        rng, rollout_rng = jax.random.split(rng)
        
        # Curriculums
        vision_radius = jnp.clip(3.0 - 2.0 * (update / 1000.0), 1.0, 3.0)
        seer_entropy_coef = jnp.clip(0.1 - 0.05 * (update / 1000.0), 0.05, 0.1)
        
        # A. Collect Trajectory
        init_seer_carry = seer_carry
        init_doer_carry = doer_carry
        
        final_runner_state, trajectory_batch = generate_trajectory_and_gae(
            params,
            rollout_rng,
            env_obs,
            env_state,
            seer_carry,
            doer_carry,
            vision_radius,
            jnp.array(config["cic_coef"], dtype=jnp.float32),
            config["num_steps"],
            step_fn, critic.apply, doer.apply
        )
        
        # Extract for next loop iteration
        params, seer_carry, doer_carry, env_state, env_obs, _, _ = final_runner_state
        
        # B. Generalized Advantage Estimation (GAE)
        # GAE is now calculated inside generate_trajectory_and_gae
        
        # C. Update Networks
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        
        batched_trajectory = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1),
            trajectory_batch
        )
        batched_seer_carry = init_seer_carry
        batched_doer_carry = init_doer_carry

        actor_state, actor_metrics = update_actor(
            actor_state, batched_trajectory, batched_seer_carry, batched_doer_carry, 
            seer.apply, doer.apply, actor_rng, seer_entropy_coef
        )
        critic_state, critic_metrics = update_critic(
            critic_state, batched_trajectory, critic.apply, critic_rng
        )
        
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
                "seer_reward": trajectory_batch.reward[..., 0].mean(),
                "doer_reward": trajectory_batch.reward[..., 1].mean(),
                "task_reward": trajectory_batch.task_reward.mean(),
                "progress_reward": trajectory_batch.progress_reward.mean(),
                "follow_reward": trajectory_batch.follow_reward.mean(),
                "step_penalty_component": trajectory_batch.step_penalty_component.mean(),
                "bump_penalty_component": trajectory_batch.bump_penalty_component.mean(),
                "seer_grad_norm": actor_metrics.get("seer_grad_norm", 0.0),
                "doer_grad_norm": actor_metrics.get("doer_grad_norm", 0.0),
                "thought_variance": actor_metrics.get("thought_variance", 0.0),
                "vision_radius": vision_radius,
                "seer_entropy_coef": seer_entropy_coef,
                "doer_perception_level": config["doer_perception_level"],
                "curriculum_success_streak": success_streak,
            })
            rng, cic_rng = jax.random.split(rng)
            cic_score = compute_cic(
                doer.apply,
                params["doer"],
                trajectory_batch,
                init_doer_carry,
                cic_rng
            )
            print(
                f"Update {update}/{num_updates} | "
                f"Level: {config['doer_perception_level']} | "
                f"Seer Reward: {trajectory_batch.reward[..., 0].mean():.3f} | "
                f"Doer Reward: {trajectory_batch.reward[..., 1].mean():.3f} | "
                f"Task: {trajectory_batch.task_reward.mean():.3f} | "
                f"Progress: {trajectory_batch.progress_reward.mean():.3f} | "
                f"Follow: {trajectory_batch.follow_reward.mean():.3f} | "
                f"Step: {trajectory_batch.step_penalty_component.mean():.3f} | "
                f"Bump: {trajectory_batch.bump_penalty_component.mean():.3f} | "
                f"Seer Grad: {actor_metrics.get('seer_grad_norm', 0.0):.4f} | "
                f"Doer Grad: {actor_metrics.get('doer_grad_norm', 0.0):.4f} | "
                f"CIC: {cic_score:.3f}"
            )
            
            # Log a small sample of the discrete messages to see distribution
            sample_msgs = actor_metrics.get("discrete_messages")
            if sample_msgs is not None:
                # Shape might be flattened over sequence and batch. Let's just print the first 5 elements safely.
                flat_msgs = sample_msgs.reshape((-1, len(config["fsq_levels"])))
                print(f"Message sample: {flat_msgs[0:5]}")

        # E. Causal Influence of Communication and Heatmap logging
        if update % 50 == 0:

            
            # Compute Heatmap
            levels = np.array(config["fsq_levels"], dtype=np.int32)
            multipliers = np.ones_like(levels)
            for idx in range(len(levels) - 2, -1, -1):
                multipliers[idx] = multipliers[idx + 1] * levels[idx + 1]

            m = np.rint(np.array(trajectory_batch.message)).astype(np.int32)
            m = np.clip(m, 0, levels - 1)
            a = np.array(trajectory_batch.action)
            m_flat = (m * multipliers).sum(axis=-1).astype(np.int32).reshape(-1)
            a_flat = a.astype(np.int32).reshape(-1)
            
            num_actions = env.num_actions
            num_message_codes = int(np.prod(levels))
            H, _, _ = np.histogram2d(
                m_flat,
                a_flat,
                bins=[np.arange(num_message_codes + 1), np.arange(num_actions + 1)]
            )
            
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(5, 10))
                cax = ax.imshow(H, aspect='auto', cmap='viridis', interpolation='none')
                fig.colorbar(cax)
                ax.set_xlabel("Doer Action")
                ax.set_ylabel("Seer Message Index")
                ax.set_title(f"Signal-to-Action Heatmap (CIC: {cic_score:.3f})")
                heatmap_log = wandb.Image(fig)
                plt.close(fig)
            except ImportError:
                heatmap_log = wandb.Image(H / (H.max() + 1e-8))
                pass

            wandb.log({
                "CIC_Score": cic_score,
                "Signal_Action_Heatmap": heatmap_log
            }, commit=False)
            
            

        if update > 0 and update % config["visualize_every"] == 0:
            rng, viz_rng = jax.random.split(rng)
            viz_path = Path(config["visualize_dir"]) / f"episode_{update:05d}.gif"
            _, solved = visualize_episode(
                env,
                params,
                viz_rng,
                seer,
                doer,
                filename=str(viz_path),
                vision_radius=vision_radius,
                max_steps=config["visualize_max_steps"],
            )
            if solved:
                success_streak += 1
            else:
                success_streak = 0

            if (
                success_streak >= config["curriculum_success_streak"]
                and config["doer_perception_level"] < 3
            ):
                config["doer_perception_level"] += 1
                success_streak = 0
                env.doer_perception_level = config["doer_perception_level"]
                step_fn = make_rollout_step(
                    env,
                    seer.apply,
                    doer.apply,
                    critic.apply,
                    follow_reward_scale=config["follow_reward_scale"],
                    msg_diversity_scale=config["msg_diversity_scale"],
                )
                rng, env_rng = jax.random.split(rng, 2)
                reset_keys = jax.random.split(env_rng, config["num_envs"])
                env_obs, env_state = env.reset_batch(reset_keys, vision_radius=jnp.array(3.0))
                seer_carry = seer.initialize_carry(config["num_envs"], 128)
                doer_carry = doer.initialize_carry(config["num_envs"], 128)
                print(
                    f"Curriculum advanced to doer_perception_level="
                    f"{config['doer_perception_level']}"
                )

if __name__ == "__main__":
    main()
