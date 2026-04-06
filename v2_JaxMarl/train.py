import jax
import jax.numpy as jnp
import optax
from pathlib import Path
import flax.linen as nn
from flax.training.train_state import TrainState

# Import our custom modules
from models.seer import Seer
from models.doer import Doer
from envs.navix_wrapper import NavixGridWrapper, UNSET_POSITION
from training.loop import generate_trajectory_and_gae, make_rollout_step
from agents.mappo import update_actor, update_critic, Transition
import navix as nx
from eval.visualize import visualize_episode
import numpy as np
import wandb


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


def reset_curriculum_batch(
    env,
    rng,
    num_envs,
    vision_radius,
    control_mode,
    fixed_goal_position,
    fixed_start_position,
):
    rng, env_rng = jax.random.split(rng)
    reset_keys = jax.random.split(env_rng, num_envs)
    env_obs, env_state = env.reset_batch(
        reset_keys,
        vision_radius=vision_radius,
        control_mode=control_mode,
        fixed_goal_position=fixed_goal_position,
        fixed_start_position=fixed_start_position,
    )
    return rng, env_obs, env_state


def sample_curriculum_anchor(
    env,
    rng,
    vision_radius,
    control_mode,
    fixed_goal_position=UNSET_POSITION,
    exclude_start_position=UNSET_POSITION,
    max_attempts=512,
):
    for _ in range(max_attempts):
        rng, sample_rng = jax.random.split(rng)
        _, timestep = env.reset(
            sample_rng,
            vision_radius=vision_radius,
            control_mode=control_mode,
            fixed_goal_position=fixed_goal_position,
        )
        start_position = env.player_position(timestep)
        if jnp.any(exclude_start_position < 0) or not bool(
            jnp.all(start_position == exclude_start_position)
        ):
            return rng, env.goal_position(timestep), start_position
    raise RuntimeError(
        "Failed to sample a new curriculum start position. "
        "This curriculum requires an environment with random starts, "
        "for example 'Navix-Empty-Random-8x8-v0'."
    )


def collect_message_action_trace(
    env,
    params,
    rng,
    seer,
    doer,
    vision_radius,
    max_steps,
    control_mode,
    fixed_goal_position,
    fixed_start_position,
):
    action_labels = ("turn_left", "turn_right", "forward")
    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(
        reset_rng,
        vision_radius=vision_radius,
        control_mode=control_mode,
        fixed_goal_position=fixed_goal_position,
        fixed_start_position=fixed_start_position,
    )

    seer_carry = seer.initialize_carry(batch_size=1, hidden_size=128)
    doer_carry = doer.initialize_carry(batch_size=1, hidden_size=128)
    trace_lines = []
    done = False
    step_count = 0

    while not bool(done) and step_count < max_steps:
        global_map = obs["global_map"][None, ...]
        symbolic_state = obs["symbolic_state"][None, ...]
        local_view = obs["local_view"][None, ...]
        proprioception = obs["proprioception"][None, ...]

        seer_carry, message, _, _ = seer.apply(
            {"params": params["seer"]},
            seer_carry,
            global_map,
            symbolic_state,
        )
        doer_carry, action_logits = doer.apply(
            {"params": params["doer"]},
            doer_carry,
            local_view,
            proprioception,
            message,
        )

        message_np = np.asarray(message[0]).round(3).tolist()
        action = int(jnp.argmax(action_logits[0]))
        action_label = action_labels[action] if action < len(action_labels) else str(action)
        player_position = np.asarray(env.player_position(state)).tolist()
        trace_lines.append(
            f"t={step_count:02d} pos={player_position} msg={message_np} action={action_label}"
        )

        rng, step_rng = jax.random.split(rng)
        obs, state, _, done, info = env.step(
            step_rng,
            state,
            jnp.asarray(action, dtype=jnp.int32),
            vision_radius=vision_radius,
            control_mode=control_mode,
            fixed_goal_position=fixed_goal_position,
            fixed_start_position=fixed_start_position,
        )
        step_count += 1

    final_status = "solved" if bool(done) and float(info["task_reward"]) > 0.0 else "stopped"
    trace_lines.append(f"end={final_status} steps={step_count}")
    return rng, trace_lines


def print_communication_trace(
    env,
    params,
    rng,
    seer,
    doer,
    vision_radius,
    max_steps,
    control_mode,
    fixed_goal_position,
    fixed_start_position,
    label,
):
    rng, trace_rng = jax.random.split(rng)
    rng, trace_lines = collect_message_action_trace(
        env,
        params,
        trace_rng,
        seer,
        doer,
        vision_radius,
        max_steps,
        control_mode,
        fixed_goal_position,
        fixed_start_position,
    )
    print(f"Communication trace ({label}):")
    for line in trace_lines:
        print(line)
    return rng


def evaluate_greedy_episode(
    env,
    params,
    rng,
    seer,
    doer,
    vision_radius,
    max_steps,
    control_mode,
    fixed_goal_position,
    fixed_start_position,
):
    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(
        reset_rng,
        vision_radius=vision_radius,
        control_mode=control_mode,
        fixed_goal_position=fixed_goal_position,
        fixed_start_position=fixed_start_position,
    )

    seer_carry = seer.initialize_carry(batch_size=1, hidden_size=128)
    doer_carry = doer.initialize_carry(batch_size=1, hidden_size=128)
    done = False
    step_count = 0
    solved = False

    while not bool(done) and step_count < max_steps:
        global_map = obs["global_map"][None, ...]
        symbolic_state = obs["symbolic_state"][None, ...]
        local_view = obs["local_view"][None, ...]
        proprioception = obs["proprioception"][None, ...]

        seer_carry, message, _, seer_nav_logits = seer.apply(
            {"params": params["seer"]},
            seer_carry,
            global_map,
            symbolic_state,
        )

        if int(control_mode) == env.SEER_NAV_PHASE:
            action = jnp.argmax(seer_nav_logits[0]).astype(jnp.int32)
        else:
            doer_carry, action_logits = doer.apply(
                {"params": params["doer"]},
                doer_carry,
                local_view,
                proprioception,
                message,
            )
            action = jnp.argmax(action_logits[0]).astype(jnp.int32)

        rng, step_rng = jax.random.split(rng)
        obs, state, _, done, info = env.step(
            step_rng,
            state,
            action,
            vision_radius=vision_radius,
            control_mode=control_mode,
            fixed_goal_position=fixed_goal_position,
            fixed_start_position=fixed_start_position,
        )
        solved = solved or bool(done) and float(info["task_reward"]) > 0.0
        step_count += 1

    return rng, solved


def flatten_message_codes(message_batch, fsq_levels):
    levels = np.asarray(fsq_levels, dtype=np.int32)
    multipliers = np.ones_like(levels)
    for idx in range(len(levels) - 2, -1, -1):
        multipliers[idx] = multipliers[idx + 1] * levels[idx + 1]

    messages = np.rint(np.asarray(message_batch)).astype(np.int32)
    messages = np.clip(messages, 0, levels - 1)
    return (messages * multipliers).sum(axis=-1).astype(np.int32).reshape(-1)


def compute_message_stats(message_batch, fsq_levels):
    message_codes = flatten_message_codes(message_batch, fsq_levels)
    num_codes = int(np.prod(np.asarray(fsq_levels, dtype=np.int32)))
    counts = np.bincount(message_codes, minlength=num_codes).astype(np.float32)
    total = max(int(counts.sum()), 1)
    probs = counts / float(total)
    nonzero_probs = probs[probs > 0.0]
    entropy = float(-(nonzero_probs * np.log(nonzero_probs)).sum()) if nonzero_probs.size else 0.0
    max_entropy = float(np.log(num_codes)) if num_codes > 1 else 0.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0.0 else 0.0
    unique_codes = int((counts > 0).sum())
    return {
        "message_codes": message_codes,
        "message_code_probs": probs,
        "rollout_message_entropy": entropy,
        "rollout_message_entropy_normalized": normalized_entropy,
        "rollout_message_unique_codes": unique_codes,
        "rollout_message_num_codes": num_codes,
    }


def log_curriculum_visualization(
    env,
    params,
    rng,
    seer,
    doer,
    config,
    update,
    phase_label,
    vision_radius,
    control_mode,
    fixed_goal_position,
    fixed_start_position,
):
    viz_path = Path(config["visualize_dir"]) / f"{phase_label}_{update:05d}.gif"
    output_path, solved = visualize_episode(
        env,
        params,
        rng,
        seer,
        doer,
        filename=str(viz_path),
        vision_radius=vision_radius,
        max_steps=config["visualize_max_steps"],
        control_mode=control_mode,
        fixed_goal_position=fixed_goal_position,
        fixed_start_position=fixed_start_position,
    )
    wandb.log(
        {
            "curriculum_reset_episode": wandb.Video(str(output_path), format="gif"),
            "curriculum_reset_episode_solved": int(solved),
        },
        commit=False,
    )


def main():
    # 1. Configuration and Logging
    config = {
        "learning_rate": 3e-4,
        "num_envs": 16,
        "num_steps": 128,
        "total_timesteps": 10_000_000,
        "env_id": "Navix-Empty-Random-8x8-v0",
        "fsq_levels": [4], # Defines the categorical hypercube
        "seed": 42,
        "follow_reward_scale": 0.1,
        "progress_reward_scale": 0.2,
        "cic_coef": 0.01,
        "doer_perception_level": 2,
        "max_doer_perception_level": 3,
        "curriculum_success_streak": 3,
        "curriculum_eval_every": 25,
        "use_seer_nav_phase": True,
        "seer_required_start_positions": 5,
        "communication_start_positions_per_level": 5,
        "release_goal_after_max_level": True,
        "min_start_distance": 1.0,
        "step_penalty": 0.03,
        "bump_penalty": 0.1,
        "visualize_max_steps": 30,
        "visualize_dir": "artifacts/episodes",
    }
    
    wandb.init(entity="eleftheriaklk-ucl", project="brian_test", config=config)

    print(f"backend: {jax.default_backend()}")
    print(f"devices: {jax.devices()}")
    
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
    seer_nav_mode = jnp.array(env.SEER_NAV_PHASE, dtype=jnp.int32)
    communication_mode = jnp.array(env.COMMUNICATION_PHASE, dtype=jnp.int32)
    control_mode = (
        seer_nav_mode
        if config["use_seer_nav_phase"]
        else communication_mode
    )

    # 4. Initial Environment Reset
    fixed_goal_position = UNSET_POSITION
    fixed_start_position = UNSET_POSITION
    rng, fixed_goal_position, fixed_start_position = sample_curriculum_anchor(
        env,
        rng,
        vision_radius=jnp.array(3.0),
        control_mode=control_mode,
    )
    env.doer_perception_level = config["doer_perception_level"]
    rng, env_obs, env_state = reset_curriculum_batch(
        env,
        rng,
        config["num_envs"],
        vision_radius=jnp.array(3.0),
        control_mode=control_mode,
        fixed_goal_position=fixed_goal_position,
        fixed_start_position=fixed_start_position,
    )

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

    seer_carry = seer.initialize_carry(config["num_envs"], 128)
    doer_carry = doer.initialize_carry(config["num_envs"], 128)

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

    step_fn = make_rollout_step(
        env,
        seer.apply,
        doer.apply,
        critic.apply,
        follow_reward_scale=config["follow_reward_scale"],
    )
    current_start_success_streak = 0
    seer_mastered_starts = 0
    communication_mastered_starts = 0
    goal_randomization_enabled = False

    # 8. The Main Training Loop
    num_updates = config["total_timesteps"] // (config["num_steps"] * config["num_envs"])
    
    print(
        "Starting training... "
        f"(phase={'seer_nav' if config['use_seer_nav_phase'] else 'communication'}, "
        f"doer_perception_level={config['doer_perception_level']})"
    )
    for update in range(num_updates):
        rng, rollout_rng = jax.random.split(rng)
        
        # Curriculums
        vision_radius = jnp.clip(3.0 - 2.0 * (update / 1000.0), 1.0, 3.0)
        seer_entropy_coef = jnp.clip(0.1 - 0.09 * (update / 1000.0), 0.01, 0.1)
        
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
            control_mode,
            fixed_goal_position,
            fixed_start_position,
            jnp.array(config["cic_coef"], dtype=jnp.float32),
            config["num_steps"],
            step_fn, critic.apply, doer.apply
        )
        
        # Extract for next loop iteration
        params, seer_carry, doer_carry, env_state, env_obs, _, _, _, _, _ = final_runner_state
        
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
            seer.apply,
            doer.apply,
            actor_rng,
            control_mode,
            tuple(config["fsq_levels"]),
            seer_entropy_coef,
        )
        critic_state, critic_metrics = update_critic(
            critic_state, batched_trajectory, critic.apply, critic_rng
        )
        
        # Sync updated parameters back to the params dictionary for the next rollout
        params["seer"] = actor_state.params["seer"]
        params["doer"] = actor_state.params["doer"]
        params["critic"] = critic_state.params

        success_events = jnp.logical_and(
            trajectory_batch.done,
            trajectory_batch.task_reward > 0.0,
        )
        completed_episodes = trajectory_batch.done.astype(jnp.int32).sum()
        rollout_num_successes = success_events.astype(jnp.int32).sum()
        rollout_success_rate = jnp.where(
            completed_episodes > 0,
            rollout_num_successes.astype(jnp.float32)
            / completed_episodes.astype(jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        message_stats = compute_message_stats(trajectory_batch.message, config["fsq_levels"])
        cic_score = float(trajectory_batch.cic_score.mean())
        
        # D. Logging
        if update % 10 == 0:
            if int(control_mode) == env.SEER_NAV_PHASE:
                phase_label = "seer_nav"
            elif goal_randomization_enabled:
                phase_label = "communication_random_full"
            else:
                phase_label = "communication_random_start"
            phase_indicators = {
                "phase_seer_nav": int(phase_label == "seer_nav"),
                "phase_communication_random_start": int(
                    phase_label == "communication_random_start"
                ),
                "phase_communication_random_full": int(
                    phase_label == "communication_random_full"
                ),
            }
            wandb.log({
                "phase_label": phase_label,
                "doer_perception_level": config["doer_perception_level"],
                "current_start_success_streak": current_start_success_streak,
                "rollout_success_rate": rollout_success_rate,
                "task_reward": trajectory_batch.task_reward.mean(),
                "progress_reward": trajectory_batch.progress_reward.mean(),
                "cic_score": cic_score,
                "rollout_message_entropy_normalized": message_stats["rollout_message_entropy_normalized"],
                "rollout_message_unique_codes": message_stats["rollout_message_unique_codes"],
                "critic_loss": critic_metrics.get("critic_loss", 0.0),
                "message_distribution": wandb.Histogram(
                    np.asarray(message_stats["message_codes"])
                ),
                **phase_indicators,
            })
            print(
                f"Update {update}/{num_updates} | "
                f"Phase: {phase_label} | "
                f"Level: {config['doer_perception_level']} | "
                f"Start: ({int(fixed_start_position[0])}, {int(fixed_start_position[1])}) | "
                f"Goal: ({int(fixed_goal_position[0])}, {int(fixed_goal_position[1])}) | "
                f"Streak: {current_start_success_streak} | "
                f"SuccessRate: {float(rollout_success_rate):.3f} | "
                f"Seer Reward: {trajectory_batch.reward[..., 0].mean():.3f} | "
                f"Doer Reward: {trajectory_batch.reward[..., 1].mean():.3f} | "
                f"Task: {trajectory_batch.task_reward.mean():.3f} | "
                f"Progress: {trajectory_batch.progress_reward.mean():.3f} | "
                f"Follow: {trajectory_batch.follow_reward.mean():.3f} | "
                f"MsgH: {message_stats['rollout_message_entropy_normalized']:.3f} | "
                f"MsgUsed: {message_stats['rollout_message_unique_codes']}/"
                f"{message_stats['rollout_message_num_codes']} | "
                f"Step: {trajectory_batch.step_penalty_component.mean():.3f} | "
                f"Bump: {trajectory_batch.bump_penalty_component.mean():.3f} | "
                f"Seer Grad: {actor_metrics.get('seer_grad_norm', 0.0):.4f} | "
                f"Doer Grad: {actor_metrics.get('doer_grad_norm', 0.0):.4f} | "
                f"CIC: {cic_score:.3f}"
            )
            
        if update > 0 and update % config["curriculum_eval_every"] == 0:
            rng, greedy_solved = evaluate_greedy_episode(
                env,
                params,
                rng,
                seer,
                doer,
                vision_radius,
                config["visualize_max_steps"],
                control_mode,
                fixed_goal_position,
                fixed_start_position,
            )
            current_start_success_streak = current_start_success_streak + 1 if greedy_solved else 0

            if int(control_mode) == env.SEER_NAV_PHASE:
                if current_start_success_streak >= config["curriculum_success_streak"]:
                    current_start_success_streak = 0
                    seer_mastered_starts += 1

                    if seer_mastered_starts >= config["seer_required_start_positions"]:
                        control_mode = communication_mode
                        env.doer_perception_level = config["doer_perception_level"]
                        communication_mastered_starts = 0
                        print("")
                        print("=" * 72)
                        print("NEW PHASE: communication")
                        print("=" * 72)
                        print("")
                        print("Seer navigation mastered on five starts; switching to communication phase.")
                    else:
                        print(
                            f"Seer mastered start {seer_mastered_starts}/"
                            f"{config['seer_required_start_positions']}."
                        )

                    rng, _, fixed_start_position = sample_curriculum_anchor(
                        env,
                        rng,
                        vision_radius,
                        control_mode,
                        fixed_goal_position=fixed_goal_position,
                        exclude_start_position=fixed_start_position,
                    )
                    rng, env_obs, env_state = reset_curriculum_batch(
                        env,
                        rng,
                        config["num_envs"],
                        vision_radius,
                        control_mode,
                        fixed_goal_position,
                        fixed_start_position,
                    )
                    seer_carry = seer.initialize_carry(config["num_envs"], 128)
                    doer_carry = doer.initialize_carry(config["num_envs"], 128)
                    print("")
                    print("=" * 72)
                    print(f"NEW RANDOM POSITION: {tuple(np.asarray(fixed_start_position).tolist())}")
                    print("=" * 72)
                    print("")
                    rng, viz_rng = jax.random.split(rng)
                    log_curriculum_visualization(
                        env,
                        params,
                        viz_rng,
                        seer,
                        doer,
                        config,
                        update,
                        "seer_nav_reset",
                        vision_radius,
                        control_mode,
                        fixed_goal_position,
                        fixed_start_position,
                    )

            elif int(control_mode) == env.COMMUNICATION_PHASE:
                if current_start_success_streak >= config["curriculum_success_streak"]:
                    current_start_success_streak = 0
                    communication_mastered_starts += 1

                    if (
                        communication_mastered_starts
                        >= config["communication_start_positions_per_level"]
                    ):
                        communication_mastered_starts = 0
                        mastered_sublevel = config["doer_perception_level"]
                        rng = print_communication_trace(
                            env,
                            params,
                            rng,
                            seer,
                            doer,
                            vision_radius,
                            config["visualize_max_steps"],
                            control_mode,
                            fixed_goal_position,
                            fixed_start_position,
                            f"mastered_level_{mastered_sublevel}",
                        )
                        if config["doer_perception_level"] < config["max_doer_perception_level"]:
                            config["doer_perception_level"] += 1
                            env.doer_perception_level = config["doer_perception_level"]
                            print("")
                            print("=" * 72)
                            print(
                                "NEW DOER PERCEPTION LEVEL: "
                                f"{config['doer_perception_level']}"
                            )
                            print("=" * 72)
                            print("")
                            print(
                                f"Curriculum advanced to doer_perception_level="
                                f"{config['doer_perception_level']}"
                            )
                        else:
                            if (
                                config["release_goal_after_max_level"]
                                and not goal_randomization_enabled
                            ):
                                goal_randomization_enabled = True
                                fixed_goal_position = UNSET_POSITION
                                fixed_start_position = UNSET_POSITION
                                print("")
                                print("=" * 72)
                                print("NEW SUBCASE: communication_random_full")
                                print("=" * 72)
                                print("")
                                print(
                                    "Max doer perception level mastered; releasing fixed goal "
                                    "and continuing in fully random Empty-Random-8x8."
                                )
                            else:
                                print("Max doer perception level reached; continuing with new starts.")

                    if goal_randomization_enabled:
                        previous_goal_position = fixed_goal_position
                        rng, fixed_goal_position, fixed_start_position = sample_curriculum_anchor(
                            env,
                            rng,
                            vision_radius,
                            control_mode,
                        )
                        if not np.array_equal(
                            np.asarray(previous_goal_position),
                            np.asarray(fixed_goal_position),
                        ):
                            print("")
                            print("=" * 72)
                            print(f"NEW RANDOM GOAL: {tuple(np.asarray(fixed_goal_position).tolist())}")
                            print("=" * 72)
                            print("")
                    else:
                        rng, _, fixed_start_position = sample_curriculum_anchor(
                            env,
                            rng,
                            vision_radius,
                            control_mode,
                            fixed_goal_position=fixed_goal_position,
                            exclude_start_position=fixed_start_position,
                        )
                    rng, env_obs, env_state = reset_curriculum_batch(
                        env,
                        rng,
                        config["num_envs"],
                        vision_radius,
                        control_mode,
                        fixed_goal_position,
                        fixed_start_position,
                    )
                    seer_carry = seer.initialize_carry(config["num_envs"], 128)
                    doer_carry = doer.initialize_carry(config["num_envs"], 128)
                    print("")
                    print("=" * 72)
                    print(f"NEW RANDOM POSITION: {tuple(np.asarray(fixed_start_position).tolist())}")
                    print("=" * 72)
                    print("")
                    rng, viz_rng = jax.random.split(rng)
                    phase_label = (
                        "communication_random_full_reset"
                        if goal_randomization_enabled
                        else "communication_random_start_reset"
                    )
                    log_curriculum_visualization(
                        env,
                        params,
                        viz_rng,
                        seer,
                        doer,
                        config,
                        update,
                        phase_label,
                        vision_radius,
                        control_mode,
                        fixed_goal_position,
                        fixed_start_position,
                    )

if __name__ == "__main__":
    main()
