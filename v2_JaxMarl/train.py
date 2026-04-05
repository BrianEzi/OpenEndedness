import jax
import jax.numpy as jnp
import optax
from pathlib import Path
import flax.linen as nn
from flax.training.train_state import TrainState
import chex
from typing import Any, Dict, Tuple

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
import matplotlib.pyplot as plt
import io

@chex.dataclass
class RunnerState:
    actor_state: TrainState
    critic_state: TrainState
    seer_carry: jnp.ndarray
    doer_carry: jnp.ndarray
    env_state: Any
    env_obs: Any
    rng: jax.random.PRNGKey
    global_layout_keys: jnp.ndarray
    update: int
    success_streak: int

# For the sake of a complete script, here is a pragmatic, standard Critic
class GlobalCritic(nn.Module):
    """Evaluates the global state to guide learning (CTDE)."""
    @nn.compact
    def __call__(self, global_map: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(features=32, kernel_size=(3, 3), kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(global_map)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=128, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        # Small scale for value head to stabilize early training
        value = nn.Dense(features=2, kernel_init=nn.initializers.orthogonal(0.01))(x)
        return value

def main():
    # 1. Configuration and Logging
    config = {
        "learning_rate": 3e-4,
        "num_envs": 256,  # Increased for massive batch size -> critic stability
        "num_steps": 128,
        "total_timesteps": 10_000_000, # Increased scale
        "env_id": "Navix-Empty-Random-8x8-v0", 
        "fsq_levels": [4],
        "seed": 42,
        "follow_reward_scale": 0.1,
        "progress_reward_scale": 0.2,
        "cic_coef": 0.0,
        "doer_perception_level": 3,
        "curriculum_success_streak": 3,
        "min_start_distance": 1.0,
        "step_penalty": 0.01,
        "bump_penalty": 0.1,
        "visualize_every": 20,
        "visualize_max_steps": 10,
        "visualize_dir": "artifacts/episodes",
        "log_every": 1,
        "log_distribution_every": 5,
        "log_heatmap_every": 20,
        "log_cic_every": 10,
    }
    
    wandb.init(entity="eleftheriaklk-ucl", project="brian_test_optimized", config=config)
    
    # 2. PRNG Key Initialization
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
    reset_rng, env_rng = jax.random.split(env_rng, 2)
    # initial global_layout_key
    global_layout_key, env_rng = jax.random.split(env_rng, 2)
    # Generate exactly identical reset keys for all environments on the host-side to prevent XLA compile explosion
    # By repeating the array outside JIT, XLA doesn't try to hyper-optimize and get stuck in while-loops.
    global_layout_keys = jnp.repeat(global_layout_key[None, :], config["num_envs"], axis=0)
    env_obs, env_state = env.reset_batch(global_layout_keys, vision_radius=jnp.array(3.0))

    # 5. Network Instantiation
    seer = Seer(fsq_levels=config["fsq_levels"])
    doer = Doer(fsq_levels=config["fsq_levels"], num_actions=env.num_actions)
    critic = GlobalCritic()

    # 6. Parameter Initialization
    env_map = env_obs["global_map"][:1]
    env_sym = env_obs["symbolic_state"][:1]
    env_local = env_obs["local_view"][:1]
    env_prop = env_obs["proprioception"][:1]
    env_msg = jnp.zeros((1, len(config["fsq_levels"])))
    
    init_seer_carry_single = seer.initialize_carry(1, 128)
    init_doer_carry_single = doer.initialize_carry(1, 128)

    seer_params = seer.init(seer_init_rng, init_seer_carry_single, env_map, env_sym)["params"]
    doer_params = doer.init(doer_init_rng, init_doer_carry_single, env_local, env_prop, env_msg)["params"]
    critic_params = critic.init(critic_init_rng, env_map)["params"]

    seer_carry = seer.initialize_carry(config["num_envs"], 128)
    doer_carry = doer.initialize_carry(config["num_envs"], 128)

    # 7. Optimizer and TrainState Setup
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=config["learning_rate"], eps=1e-5)
    )

    actor_state = TrainState.create(
        apply_fn=None,
        params={"seer": seer_params, "doer": doer_params},
        tx=tx
    )
    
    critic_state = TrainState.create(
        apply_fn=critic.apply,
        params=critic_params,
        tx=tx
    )

    step_fn = make_rollout_step(
        env,
        seer.apply,
        doer.apply,
        critic.apply,
        follow_reward_scale=config["follow_reward_scale"],
    )

    # 8. Define the training step for jax.lax.scan
    def train_step(runner_state: RunnerState, _):
        # A. Curriculums
        update = runner_state.update
        # Restore constants for stability as requested by the user previously
        vision_radius = 3.0 
        seer_entropy_coef = 0.1 
        
        # B. Collect Trajectory
        rng, rollout_rng = jax.random.split(runner_state.rng)
        
        params = {
            "seer": runner_state.actor_state.params["seer"],
            "doer": runner_state.actor_state.params["doer"],
            "critic": runner_state.critic_state.params
        }
        
        final_rollout_state, trajectory_batch = generate_trajectory_and_gae(
            params,
            rollout_rng,
            runner_state.env_obs,
            runner_state.env_state,
            runner_state.seer_carry,
            runner_state.doer_carry,
            vision_radius,
            runner_state.global_layout_keys,
            jnp.array(config["cic_coef"], dtype=jnp.float32),
            config["num_steps"],
            step_fn, critic.apply, doer.apply
        )
        
        # C. Update Networks
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        
        batched_trajectory = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1),
            trajectory_batch
        )
        
        actor_state, actor_metrics = update_actor(
            runner_state.actor_state, 
            batched_trajectory, 
            runner_state.seer_carry, 
            runner_state.doer_carry, 
            seer.apply, doer.apply, actor_rng, seer_entropy_coef
        )
        critic_state, critic_metrics = update_critic(
            runner_state.critic_state, 
            batched_trajectory, 
            critic.apply, critic_rng
        )
        
        # D. Update Runner State
        _, next_seer_carry, next_doer_carry, next_env_state, next_env_obs, _, _, next_global_layout_keys = final_rollout_state
        
        next_state = RunnerState(
            actor_state=actor_state,
            critic_state=critic_state,
            seer_carry=next_seer_carry,
            doer_carry=next_doer_carry,
            env_state=next_env_state,
            env_obs=next_env_obs,
            rng=rng,
            global_layout_keys=next_global_layout_keys,
            update=update + 1,
            success_streak=runner_state.success_streak
        )
        
        metrics = {
            "update": update,
            "seer_loss": actor_metrics.get("seer_loss", 0.0),
            "doer_loss": actor_metrics.get("doer_loss", 0.0),
            "entropy": actor_metrics.get("entropy", 0.0),
            "critic_loss": critic_metrics.get("critic_loss", 0.0),
            "seer_reward": trajectory_batch.reward[..., 0].mean(),
            "doer_reward": trajectory_batch.reward[..., 1].mean(),
            "task_reward": trajectory_batch.task_reward.mean(),
            "progress_reward": trajectory_batch.progress_reward.mean(),
            "follow_reward": trajectory_batch.follow_reward.mean(),
            "vision_radius": vision_radius,
            "seer_entropy_coef": seer_entropy_coef,
            "thought_variance": actor_metrics.get("thought_variance", 0.0),
            "discrete_messages": actor_metrics.get("discrete_messages"),
            "trajectory_batch": trajectory_batch, # Partially return for CIC calculation outside scan if needed

        }
        
        return next_state, metrics

    # 9. Initial Runner State
    runner_state = RunnerState(
        actor_state=actor_state,
        critic_state=critic_state,
        seer_carry=seer_carry,
        doer_carry=doer_carry,
        env_state=env_state,
        env_obs=env_obs,
        rng=rng,
        global_layout_keys=global_layout_keys,
        update=0,
        success_streak=0
    )

    # Compile the training block
    train_block = jax.jit(lambda state: jax.lax.scan(train_step, state, None, length=config["log_every"]))

    # 10. The Main Training Loop (Python side handles logging and viz)
    total_updates = config["total_timesteps"] // (config["num_steps"] * config["num_envs"])
    num_blocks = total_updates // config["log_every"]
    
    print(f"Starting optimized training with {config['num_envs']} envs...")
    
    mastered_environments = 0
    
    for block in range(num_blocks):
        runner_state, metrics = train_block(runner_state)
        
        # Extract the last update's metrics for logging
        last_metrics = jax.tree_util.tree_map(lambda x: x[-1], metrics)
        update_idx = int(last_metrics["update"])
        
        
        # Calculate vocab_used on the host side using numpy to avoid JAX static shaping issues
        msgs = metrics["discrete_messages"]
        vocab_used = len(np.unique(np.array(msgs)))
        
        # A. Logging to WandB
        log_dict = {
            "update": update_idx,
            "seer_loss": last_metrics["seer_loss"],
            "doer_loss": last_metrics["doer_loss"],
            "entropy": last_metrics["entropy"],
            "critic_loss": last_metrics["critic_loss"],
            "seer_reward": last_metrics["seer_reward"],
            "doer_reward": last_metrics["doer_reward"],
            "task_reward": last_metrics["task_reward"],
            "progress_reward": last_metrics["progress_reward"],
            "follow_reward": last_metrics["follow_reward"],
            "seer_entropy_coef": last_metrics["seer_entropy_coef"],
            "vocab_used": vocab_used,
            "curriculum_level": env.doer_perception_level,
        }

        # B. Periodic Evaluation and Visualization
        if update_idx > 0 and update_idx % config["log_distribution_every"] == 0:
            print(f"Update {update_idx} | Seer/Doer Loss: {last_metrics['seer_loss']:.4f}/{last_metrics['doer_loss']:.4f} | Task Reward: {last_metrics['task_reward']:.4f}")
            
            # Log message distribution as a histogram
            msgs = metrics["discrete_messages"].reshape(-1) # Flatten all messages in the block
            log_dict["message_distribution"] = wandb.Histogram(np.array(msgs))

        # C. Signal-to-Action Heatmap (Auditing if discrete symbols correlate with actions)
        if update_idx > 0 and update_idx % config["log_heatmap_every"] == 0:
            traj = metrics["trajectory_batch"]
            m = np.array(traj.message).reshape((-1, len(config["fsq_levels"])))
            a = np.array(traj.action).reshape(-1)
            
            levels = np.array(config["fsq_levels"])
            if len(levels) == 1:
                m_idx = m[:, 0]
            else:
                multipliers = np.cumprod(np.concatenate([np.array([1]), levels[:-1]]))
                m_idx = (m * multipliers).sum(axis=-1)
            
            num_message_codes = int(np.prod(levels))
            num_actions = env.num_actions
            H, _, _ = np.histogram2d(
                m_idx, a, 
                bins=[np.arange(num_message_codes + 1), np.arange(num_actions + 1)]
            )
            
            # Row Normalization to show P(Action | Message)
            row_sums = H.sum(axis=1, keepdims=True)
            H_norm = np.divide(H, row_sums, out=np.zeros_like(H), where=row_sums!=0)
            
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(H_norm, cmap='viridis')
            ax.set_xticks(np.arange(num_actions))
            ax.set_yticks(np.arange(num_message_codes))
            ax.set_xticklabels(['Forward', 'Left', 'Right']) # Navix specific
            ax.set_ylabel('Message Index (FSQ Code)')
            ax.set_title('P(Action | Message)')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Log top messages representation
            top_msgs = np.argsort(row_sums.flatten())[::-1][:5]
            action_labels = ['Forward', 'Left', 'Right']
            msg_table_data = []
            for tm in top_msgs:
                if row_sums[tm][0] > 0:
                    dominant_action_idx = np.argmax(H_norm[tm])
                    dominant_action = action_labels[dominant_action_idx]
                    percentage = H_norm[tm][dominant_action_idx] * 100
                    msg_table_data.append([str(tm), f"{dominant_action} ({percentage:.1f}%)"])
            
            log_dict["top_message_meanings"] = wandb.Table(data=msg_table_data, columns=["Message ID", "Dominant Meaning"])

            from PIL import Image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            heatmap_img = Image.open(buf)
            log_dict["signal_action_heatmap"] = wandb.Image(heatmap_img)
            plt.close(fig)

        if update_idx > 0 and update_idx % config["visualize_every"] == 0:
            params = {
                "seer": runner_state.actor_state.params["seer"],
                "doer": runner_state.actor_state.params["doer"],
                "critic": runner_state.critic_state.params
            }
            rng, viz_rng = jax.random.split(runner_state.rng)
            viz_path = Path(config["visualize_dir"]) / f"episode_{update_idx:05d}.gif"
            _, solved = visualize_episode(
                env, params, viz_rng, seer, doer,
                filename=str(viz_path),
                vision_radius=3.0,
                max_steps=config["visualize_max_steps"],
            )
            
            # Log Video to WandB
            log_dict.update({
                "episode_viz": wandb.Video(str(viz_path), fps=4, format="gif"),
                "episode_solved": float(solved)
            })

        if update_idx > 0 and update_idx % config["log_cic_every"] == 0:
            # Compute and log CIC score (Seer Score)
            rng, cic_rng = jax.random.split(runner_state.rng)
            last_traj = jax.tree_util.tree_map(lambda x: x[-1], metrics["trajectory_batch"])
            cic_score = compute_cic(
                doer.apply, runner_state.actor_state.params["doer"], last_traj, runner_state.doer_carry, cic_rng
            )
            log_dict["CIC_Score"] = cic_score

        # D. Curriculum Check (Decoupled from visualization)
        # Average number of maze completions per environment during this block
        traj_batch = metrics["trajectory_batch"]
        completions_per_env = np.sum(traj_batch.task_reward) / config["num_envs"]
        
        # Single-Environment Mastery Check
        if completions_per_env >= 0.9:
            mastered_environments += 1
            print(f"Mastered layout! Switching to new layout (Total mastered: {mastered_environments})")
            
            # Re-roll a single new map for ALL environments
            rng = runner_state.rng
            rng, new_seed_rng = jax.random.split(rng)
            new_global_layout_key, _ = jax.random.split(new_seed_rng)
            
            # Reset all current envs on the new layout
            new_global_layout_keys = jnp.repeat(new_global_layout_key[None, :], config["num_envs"], axis=0)
            new_obs, new_state = env.reset_batch(new_global_layout_keys, vision_radius=jnp.array(3.0))
            
            # Immediately overwrite the state locally
            runner_state = runner_state.replace(
                env_obs=new_obs,
                env_state=new_state,
                global_layout_keys=new_global_layout_keys,
                rng=rng,
                success_streak=runner_state.success_streak + 1
            )
        else:
            runner_state = runner_state.replace(success_streak=0)
            
        log_dict["success_streak"] = runner_state.success_streak
        log_dict["completions_per_env"] = completions_per_env
        log_dict["mastered_environments"] = mastered_environments

        # Finally, upload everything!
        wandb.log(log_dict)

        # Manual Level advancement logic
        if runner_state.success_streak >= config["curriculum_success_streak"] and env.doer_perception_level < 3:
            env.doer_perception_level += 1
            curr_level = env.doer_perception_level
            print(f"Advancing to perception level {curr_level}")
            
            # Re-compile step_fn with new env setting (or make level a dyn param if preferred)
            step_fn = make_rollout_step(env, seer.apply, doer.apply, critic.apply, follow_reward_scale=config["follow_reward_scale"])
            train_block = jax.jit(lambda state: jax.lax.scan(train_step, state, None, length=config["log_every"]))
            
            # Reset envs for the new level
            rng, env_rng = jax.random.split(runner_state.rng, 2)
            reset_keys = jax.random.split(env_rng, config["num_envs"])
            new_obs, new_state = env.reset_batch(reset_keys, vision_radius=jnp.array(3.0))
            runner_state = runner_state.replace(env_obs=new_obs, env_state=new_state, rng=rng)

if __name__ == "__main__":
    main()
