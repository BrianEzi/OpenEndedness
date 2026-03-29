# Entry point for solo Seer pre-training.
# The Seer is trained alone to solve the navigation task before being paired with the Doer.
# Once trained, its LSTM weights are transferred to the full Seer+Doer system.

import jax
import jax.numpy as jnp
import optax
from pathlib import Path
from flax.training.train_state import TrainState
from models.seer_solo import SeerSolo
from envs.navix_wrapper import NavixGridWrapper
from training.seer_solo import (
    make_seer_solo_rollout_step,
    generate_seer_solo_trajectory,
    update_seer_solo,
)
import navix as nx
import wandb
import pickle

def main():
    config = {
        "learning_rate": 3e-4,
        "num_envs": 16,
        "num_steps": 128,
        "total_timesteps": 5_000_000,
        "env_id": "Navix-Empty-8x8-v0",
        "seed": 42,
        "progress_reward_scale": 0.2,
        "step_penalty": 0.01,
        "bump_penalty": 0.1,
        "min_start_distance": 1.0,
        "visualize_every": 400,
        "visualize_max_steps": 30,
        "visualize_dir": "artifacts/episodes",
        "entropy_coef": 0.05,
        "checkpoint_dir": "artifacts/checkpoints",
    }

    wandb.init(entity="eleftheriaklk-ucl", project="efi_test", config=config)

    rng = jax.random.PRNGKey(config["seed"])
    rng, seer_init_rng, env_rng = jax.random.split(rng, 3)

    # Environment
    raw_env = nx.make(config["env_id"])
    env = NavixGridWrapper(
        raw_env,
        progress_reward_scale=config["progress_reward_scale"],
        step_penalty=config["step_penalty"],
        bump_penalty=config["bump_penalty"],
        min_start_distance=config["min_start_distance"],
        doer_perception_level=0,
    )

    # Initial reset
    reset_keys = jax.random.split(env_rng, config["num_envs"])
    env_obs, env_state = env.reset_batch(reset_keys)

    # Network
    seer = SeerSolo()
    dummy_map = env_obs["global_map"][:1]
    dummy_sym = env_obs["symbolic_state"][:1]
    init_carry = seer.initialize_carry(1, 128)
    seer_params = seer.init(seer_init_rng, init_carry, dummy_map, dummy_sym)["params"]
    seer_carry = seer.initialize_carry(config["num_envs"], 128)

    # Optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=config["learning_rate"], eps=1e-5),
    )
    actor_state = TrainState.create(apply_fn=None, params=seer_params, tx=tx)

    # Training loop
    step_fn = make_seer_solo_rollout_step(env, seer.apply)
    num_updates = config["total_timesteps"] // (config["num_steps"] * config["num_envs"])

    print(f"Starting solo Seer training... ({num_updates} updates)")

    for update in range(num_updates):
        rng, rollout_rng = jax.random.split(rng)
        init_seer_carry = seer_carry

        final_runner_state, trajectory = generate_seer_solo_trajectory(
            actor_state.params, rollout_rng, env_obs, env_state,
            seer_carry, config["num_steps"], step_fn
        )

        _, seer_carry, env_state, env_obs, _ = final_runner_state

        actor_state, metrics = update_seer_solo(
        actor_state, trajectory, init_seer_carry, seer.apply, rng,
        entropy_coef=config["entropy_coef"],
    )


        if update % 10 == 0:
            _, _, _, _, reward, done, _, _ = trajectory
            wandb.log({
                "update": update,
                "reward": reward.mean(),
                "done_rate": done.mean(),
                "actor_loss": metrics["actor_loss"],
                "entropy": metrics["entropy"],
            })

            print(
                f"Update {update}/{num_updates} | "
                f"Reward: {reward.mean():.3f} | "
                f"Loss: {metrics['actor_loss']:.3f} | "
                f"Entropy: {metrics['entropy']:.3f}"
            )
    ckpt_dir = Path(config["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "seer_solo_final.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump(jax.device_get(actor_state.params), f)
    print(f"Saved SeerSolo checkpoint to {ckpt_path}")
    wandb.save(str(ckpt_path))


if __name__ == "__main__":
    main()
