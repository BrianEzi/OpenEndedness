from pathlib import Path

import jax
import jax.numpy as jnp
from PIL import Image

from envs.navix_wrapper import UNSET_POSITION


def visualize_episode(
    env,
    params,
    rng,
    seer,
    doer,
    filename="episode.gif",
    vision_radius=jnp.array(2.0),
    max_steps=200,
    control_mode=jnp.array(1, dtype=jnp.int32),
    fixed_goal_position=UNSET_POSITION,
    fixed_start_position=UNSET_POSITION,
):
    """Run one greedy evaluation episode and save it as a GIF."""
    frames = []
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
    solved = False
    step_count = 0

    while not bool(done) and step_count < max_steps:
        frame = env.render(state, control_mode=int(control_mode))
        frames.append(Image.fromarray(frame))

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
        obs, state, reward, done, info = env.step(
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

    if not frames:
        raise RuntimeError("Visualization episode produced no frames.")

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
    )
    print(f"Episode saved to {output_path}")
    return output_path, solved
