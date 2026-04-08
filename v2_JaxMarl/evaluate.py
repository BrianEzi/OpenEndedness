import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints
from PIL import Image, ImageDraw, ImageFont

from envs.two_doer_grid import TwoDoerBottleneckEnv, UNSET_TWO_DOER_POSITIONS
from models.doer import Doer
from models.seer import Seer
from training.action_masking import mask_pick_actions_until_menu_visible
from training.message_masking import hard_mask_inactive_message_bits


NAV_ACTION_LABELS = ("stay", "up", "right", "down", "left")
DOER_KEYS = ("doer_a", "doer_b")
COLOR_NAMES = ("red", "green", "blue", "yellow")
SHAPE_NAMES = ("solid_square", "plus", "x", "frame")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained two-doer checkpoint by visualizing greedy episodes and "
            "analyzing the communication protocol."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="jonny-vr/two-doers-one-seer-curriculum",
        help=(
            "Local checkpoint path or a Hugging Face repo id. "
            "If a repo id is given, the script will try to download it via huggingface_hub."
        ),
    )
    parser.add_argument(
        "--fsq-levels",
        type=str,
        default="2,2,2,2",
        help="Comma-separated FSQ levels used by the checkpoint, e.g. 2,2,2,2 or 4,4.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/eval",
        help="Directory where GIFs, traces, and JSON reports are written.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=12,
        help="How many greedy episodes to roll out for empirical analysis.",
    )
    parser.add_argument(
        "--num-visualizations",
        type=int,
        default=3,
        help="How many greedy episodes to save as annotated GIFs.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=2,
        choices=(1, 2, 3),
        help="How many consecutive messages to use in the controlled codebook probe.",
    )
    parser.add_argument(
        "--distractor-packs",
        type=int,
        default=4,
        help="How many distractor menu variants to test per target object.",
    )
    parser.add_argument(
        "--selection-phase-level",
        type=int,
        default=2,
        choices=(1, 2),
        help="Environment phase for greedy policy evaluation. Use 2 for the full task.",
    )
    parser.add_argument(
        "--doer-perception-level",
        type=int,
        default=2,
        choices=(2, 3),
        help="Doer perception level to evaluate under.",
    )
    parser.add_argument("--grid-height", type=int, default=10)
    parser.add_argument("--grid-width", type=int, default=12)
    parser.add_argument("--local-view-size", type=int, default=3)
    parser.add_argument("--corridor-length", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=48)
    parser.add_argument("--pick-object-max-steps", type=int, default=8)
    parser.add_argument("--pick-object-listen-steps", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--print-summary-limit",
        type=int,
        default=12,
        help="Maximum number of top empirical message-pair summaries to print per doer.",
    )
    return parser.parse_args()


def parse_fsq_levels(levels_str: str):
    return tuple(int(level.strip()) for level in levels_str.split(",") if level.strip())


def item_label(item_id: int) -> str:
    color = COLOR_NAMES[item_id // 4]
    shape = SHAPE_NAMES[item_id % 4]
    return f"{color}_{shape}"


def action_label(action: int) -> str:
    if action < 5:
        return NAV_ACTION_LABELS[action]
    return f"pick_{action - 5}"


def looks_like_hf_repo(checkpoint_ref: str) -> bool:
    checkpoint_path = Path(checkpoint_ref).expanduser()
    return not checkpoint_path.exists() and "/" in checkpoint_ref and not checkpoint_ref.startswith(".")


def resolve_checkpoint_reference(checkpoint_ref: str) -> Path:
    checkpoint_path = Path(checkpoint_ref).expanduser()
    if checkpoint_path.exists():
        return checkpoint_path.resolve()

    if not looks_like_hf_repo(checkpoint_ref):
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_ref}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Checkpoint looks like a Hugging Face repo id, but huggingface_hub is not installed. "
            "Install it with `pip install huggingface_hub` or pass a local checkpoint path."
        ) from exc

    local_dir = snapshot_download(repo_id=checkpoint_ref)
    return Path(local_dir).resolve()


def checkpoint_step_from_name(path: Path):
    if not path.name.startswith("checkpoint_"):
        return None
    suffix = path.name.split("_")[-1]
    if not suffix.isdigit():
        return None
    return int(suffix)


def is_orbax_checkpoint_dir(path: Path) -> bool:
    return path.is_dir() and (path / "_CHECKPOINT_METADATA").exists()


def find_checkpoint_path(checkpoint_path: Path) -> Path:
    direct_step = checkpoint_step_from_name(checkpoint_path)
    if direct_step is not None:
        return checkpoint_path

    if checkpoint_path.is_dir():
        # Direct Orbax OCDBT checkpoint (e.g. downloaded from HuggingFace without a
        # checkpoint_N wrapper directory).
        if is_orbax_checkpoint_dir(checkpoint_path):
            return checkpoint_path

        candidates = []
        for candidate in checkpoint_path.rglob("checkpoint_*"):
            step = checkpoint_step_from_name(candidate)
            if step is not None:
                candidates.append((step, candidate))
            elif is_orbax_checkpoint_dir(candidate):
                candidates.append((0, candidate))
        if candidates:
            candidates.sort(key=lambda item: (item[0], str(item[1])))
            return candidates[-1][1]

    raise FileNotFoundError(
        "Could not find any Flax checkpoint matching `checkpoint_*` under "
        f"{checkpoint_path}"
    )


def resolve_checkpoint_location(checkpoint_path: Path):
    concrete_path = find_checkpoint_path(checkpoint_path.resolve())
    step = checkpoint_step_from_name(concrete_path)
    # step is None for Orbax checkpoints not following the checkpoint_N naming convention
    return concrete_path.parent, step, concrete_path


def _restore_orbax(checkpoint_path: Path):
    try:
        import orbax.checkpoint as ocp
    except ImportError as exc:
        raise ImportError(
            "Orbax checkpoint detected but orbax is not installed. "
            "Install it with `pip install orbax-checkpoint`."
        ) from exc
    checkpointer = ocp.PyTreeCheckpointer()
    return checkpointer.restore(str(checkpoint_path))


def restore_params(checkpoint_ref: str):
    resolved_path = resolve_checkpoint_reference(checkpoint_ref)
    restore_dir, step, concrete_checkpoint_path = resolve_checkpoint_location(resolved_path)

    if step is None:
        # Orbax OCDBT checkpoint
        params = _restore_orbax(concrete_checkpoint_path)
    else:
        params = checkpoints.restore_checkpoint(
            ckpt_dir=str(restore_dir),
            target=None,
            step=step,
            prefix="checkpoint_",
        )

    if params is None:
        raise FileNotFoundError(
            "Checkpoint restore returned None. "
            f"Resolved repo/path: {resolved_path}. "
            f"Concrete checkpoint candidate: {concrete_checkpoint_path}."
        )
    if "seer" not in params or "doer" not in params:
        raise KeyError("Checkpoint must contain both `seer` and `doer` parameters.")
    return params, concrete_checkpoint_path


def infer_message_dim_from_params(params) -> int:
    try:
        return int(params["doer"]["Dense_0"]["kernel"].shape[0])
    except KeyError as exc:
        raise KeyError(
            "Could not infer message dimension from checkpoint params at "
            "params['doer']['Dense_0']['kernel']."
        ) from exc


def build_env(args, selection_phase_level=None):
    return TwoDoerBottleneckEnv(
        grid_height=args.grid_height,
        grid_width=args.grid_width,
        local_view_size=args.local_view_size,
        corridor_length=args.corridor_length,
        max_steps=args.max_steps,
        doer_perception_level=args.doer_perception_level,
        selection_phase_level=(
            args.selection_phase_level if selection_phase_level is None else selection_phase_level
        ),
        pick_object_max_steps=args.pick_object_max_steps,
        pick_object_listen_steps=args.pick_object_listen_steps,
    )


def initialize_two_doer_carry(doer, num_envs, num_doers, hidden_size):
    flat_carry = doer.initialize_carry(batch_size=num_envs * num_doers, hidden_size=hidden_size)
    return jax.tree_util.tree_map(
        lambda x: x.reshape((num_envs, num_doers, hidden_size)),
        flat_carry,
    )


def message_to_tuple(message_values):
    return tuple(int(v) for v in np.rint(np.asarray(message_values, dtype=np.float32)).astype(np.int32))


def message_key(message_values):
    return str(list(message_to_tuple(message_values)))


def message_sequence_key(message_sequence):
    return " -> ".join(str(list(word)) for word in message_sequence)


def menu_labels(menu_ids):
    return [item_label(int(item_id)) for item_id in menu_ids]


def target_slot(menu_ids, target_id):
    for idx, item_id in enumerate(menu_ids):
        if int(item_id) == int(target_id):
            return idx
    return None


def build_step_record(step_count, state, obs, messages, actions):
    positions = np.asarray(state.positions, dtype=np.int32).tolist()
    goals = np.asarray(state.goals, dtype=np.int32).tolist()
    target_items = np.asarray(state.target_items, dtype=np.int32).tolist()
    menu_ids = np.asarray(state.shuffled_menus, dtype=np.int32).tolist()
    selected_option_idx = np.asarray(state.selected_option_idx, dtype=np.int32).tolist()
    selection_attempts = np.asarray(state.selection_attempts, dtype=np.int32).tolist()
    has_selected = np.asarray(state.has_selected, dtype=bool).tolist()
    has_arrived = np.asarray(state.has_arrived, dtype=bool).tolist()
    pick_available = np.asarray(obs["pick_available"], dtype=bool).tolist()
    message_list = [list(message_to_tuple(values)) for values in np.asarray(messages[0], dtype=np.float32)]
    action_list = [int(action) for action in np.asarray(actions, dtype=np.int32).tolist()]

    chosen_items = []
    for agent_idx, action in enumerate(action_list):
        if action < 5:
            chosen_items.append(None)
        else:
            chosen_items.append(int(menu_ids[agent_idx][action - 5]))

    return {
        "step": int(step_count),
        "positions": positions,
        "goals": goals,
        "at_goal": [positions[idx] == goals[idx] for idx in range(len(positions))],
        "target_items": target_items,
        "target_labels": [item_label(item_id) for item_id in target_items],
        "target_slots": [target_slot(menu_ids[idx], target_items[idx]) for idx in range(len(target_items))],
        "menu_ids": menu_ids,
        "menu_labels": [menu_labels(agent_menu) for agent_menu in menu_ids],
        "messages": message_list,
        "actions": action_list,
        "action_labels": [action_label(action) for action in action_list],
        "chosen_items": chosen_items,
        "chosen_item_labels": [
            item_label(item_id) if item_id is not None else None for item_id in chosen_items
        ],
        "selected_option_idx": selected_option_idx,
        "selection_attempts": selection_attempts,
        "has_selected": has_selected,
        "has_arrived": has_arrived,
        "pick_available": pick_available,
    }


def build_step_annotation(step_record, agent_idx):
    side = "A" if agent_idx == 0 else "B"
    target_slot_value = step_record["target_slots"][agent_idx]
    target_slot_text = "none" if target_slot_value is None else str(target_slot_value)
    lines = [
        f"t={step_record['step']:02d}",
        f"doer={side}",
        f"pos={step_record['positions'][agent_idx]}",
        f"goal={step_record['goals'][agent_idx]}",
        f"target={step_record['target_labels'][agent_idx]}",
        f"target_slot={target_slot_text}",
        f"msg={step_record['messages'][agent_idx]}",
        f"act={step_record['action_labels'][agent_idx]}",
        f"pick_avail={int(step_record['pick_available'][agent_idx])}",
        f"at_goal={int(step_record['at_goal'][agent_idx])}",
        f"selected={int(step_record['has_selected'][agent_idx])}",
    ]
    return "\n".join(lines)


def annotate_two_doer_frame(frame, left_text, right_text):
    panel_width = 250
    background = (246, 244, 238)
    border = (190, 186, 176)
    text_color = (25, 25, 28)
    font = ImageFont.load_default()
    grid_image = Image.fromarray(frame)
    canvas = Image.new(
        "RGB",
        (grid_image.width + 2 * panel_width, grid_image.height),
        background,
    )
    canvas.paste(grid_image, (panel_width, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, panel_width - 1, grid_image.height - 1), outline=border, width=1)
    draw.rectangle(
        (panel_width + grid_image.width, 0, canvas.width - 1, grid_image.height - 1),
        outline=border,
        width=1,
    )
    draw.multiline_text((12, 12), left_text, fill=text_color, font=font, spacing=4)
    draw.multiline_text(
        (panel_width + grid_image.width + 12, 12),
        right_text,
        fill=text_color,
        font=font,
        spacing=4,
    )
    return canvas


def write_gif(frames, output_path: Path):
    if not frames:
        raise RuntimeError("Cannot save an empty GIF.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=150,
        loop=0,
    )


def run_policy_episode(
    env,
    params,
    rng,
    seer,
    doer,
    hidden_size,
    max_steps,
    fixed_positions=UNSET_TWO_DOER_POSITIONS,
    gif_path: Path | None = None,
):
    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng, fixed_positions=fixed_positions)

    seer_carry = seer.initialize_carry(batch_size=1, hidden_size=hidden_size)
    doer_carry = initialize_two_doer_carry(
        doer,
        num_envs=1,
        num_doers=env.num_doers,
        hidden_size=hidden_size,
    )

    done = False
    success = False
    step_count = 0
    frames = []
    steps = []
    final_info = {}
    final_state = state

    while not bool(done) and step_count < max_steps:
        global_map = obs["global_map"][None, ...]
        symbolic_state = obs["symbolic_state"][None, ...]
        local_views = obs["local_views"][None, ...]
        proprioceptions = obs["proprioceptions"][None, ...]
        target_images = obs["target_images"][None, ...]
        menu_images = obs["menu_images"][None, ...]

        seer_carry, messages, _, _ = seer.apply(
            {"params": params["seer"]},
            seer_carry,
            global_map,
            symbolic_state,
            target_images,
        )
        messages = hard_mask_inactive_message_bits(messages, active_bits=env.active_message_bits)

        batch_size, num_doers = local_views.shape[:2]
        flat_local_views = local_views.reshape((batch_size * num_doers,) + local_views.shape[2:])
        flat_proprioceptions = proprioceptions.reshape(
            (batch_size * num_doers,) + proprioceptions.shape[2:]
        )
        flat_messages = messages.reshape((batch_size * num_doers,) + messages.shape[2:])
        flat_doer_carry = jax.tree_util.tree_map(
            lambda x: x.reshape((batch_size * num_doers,) + x.shape[2:]),
            doer_carry,
        )
        next_flat_doer_carry, flat_logits = doer.apply(
            {"params": params["doer"]},
            flat_doer_carry,
            flat_local_views,
            flat_proprioceptions,
            flat_messages,
            menu_images.reshape((batch_size * num_doers,) + menu_images.shape[2:]),
        )
        doer_carry = jax.tree_util.tree_map(
            lambda x: x.reshape((batch_size, num_doers) + x.shape[1:]),
            next_flat_doer_carry,
        )
        masked_logits = mask_pick_actions_until_menu_visible(
            flat_logits.reshape((batch_size, num_doers, flat_logits.shape[-1])),
            menu_images,
            pick_only_phase=env.is_pick_object_phase,
            pick_available=obs["pick_available"][None, ...],
        )
        actions = jnp.argmax(masked_logits[0], axis=-1).astype(jnp.int32)

        step_record = build_step_record(step_count, state, obs, messages, actions)

        rng, step_rng = jax.random.split(rng)
        next_obs, next_state, reward, done, info = env.step(
            step_rng,
            state,
            actions,
            fixed_positions=fixed_positions,
        )

        step_record.update(
            {
                "reward": float(reward),
                "task_reward": float(info["task_reward"]),
                "individual_selection_reward": float(info["individual_selection_reward"]),
                "wrong_selection_penalty": float(info["wrong_selection_penalty"]),
                "valid_selection_count": float(info["valid_selection_count"]),
                "correct_selection_count": float(info["correct_selection_count"]),
                "progress_reward_per_doer": np.asarray(
                    info["progress_reward_per_doer"], dtype=np.float32
                ).tolist(),
                "step_penalty": float(info["step_penalty"]),
                "wall_penalty": float(info["wall_penalty"]),
                "collision_penalty": float(info["collision_penalty"]),
                "goal_distance": np.asarray(info["goal_distance"], dtype=np.float32).tolist(),
                "success_after_step": bool(info["success"]),
                "eventual_success_after_step": bool(info["eventual_success"]),
                "first_try_success_after_step": bool(info["first_try_success"]),
                "failed_after_step": bool(info["failed"]),
            }
        )
        steps.append(step_record)

        if gif_path is not None:
            left_text = build_step_annotation(step_record, 0)
            right_text = build_step_annotation(step_record, 1)
            frames.append(annotate_two_doer_frame(env.render(state), left_text, right_text))

        success = success or bool(info["success"])
        final_info = info
        final_state = next_state
        obs = next_obs
        state = next_state
        step_count += 1

    if gif_path is not None and steps:
        last_step = dict(steps[-1])
        last_step["step"] = int(step_count)
        left_text = build_step_annotation(last_step, 0) + "\nfinal=1"
        right_text = build_step_annotation(last_step, 1) + "\nfinal=1"
        frames.append(annotate_two_doer_frame(env.render(final_state), left_text, right_text))
        write_gif(frames, gif_path)

    episode_record = {
        "success": bool(success),
        "num_steps": int(step_count),
        "phase_name": env.phase_name,
        "steps": steps,
        "final_positions": np.asarray(final_state.positions, dtype=np.int32).tolist(),
        "final_goals": np.asarray(final_state.goals, dtype=np.int32).tolist(),
        "final_selected_correctly": np.asarray(final_state.selected_correctly, dtype=bool).tolist(),
        "final_first_selection_correct": np.asarray(
            final_state.first_selection_correct, dtype=bool
        ).tolist(),
        "final_has_selected": np.asarray(final_state.has_selected, dtype=bool).tolist(),
        "final_selection_attempts": np.asarray(final_state.selection_attempts, dtype=np.int32).tolist(),
        "final_info": {
            "success": bool(final_info.get("success", False)),
            "eventual_success": bool(final_info.get("eventual_success", False)),
            "first_try_success": bool(final_info.get("first_try_success", False)),
            "failed": bool(final_info.get("failed", False)),
        },
    }
    if gif_path is not None:
        episode_record["gif_path"] = str(gif_path)
    return rng, episode_record


def all_messages(fsq_levels):
    return [tuple(word) for word in itertools.product(*[range(level) for level in fsq_levels])]


def all_message_sequences(fsq_levels, sequence_length: int):
    messages = all_messages(fsq_levels)
    return [tuple(seq) for seq in itertools.product(messages, repeat=sequence_length)]


def distractor_packs_for_target(target_id: int, num_items: int, num_packs: int):
    others = [item_id for item_id in range(num_items) if item_id != target_id]
    packs = []
    stride = max(1, len(others) // max(1, num_packs))
    for pack_idx in range(num_packs):
        start = (pack_idx * stride) % len(others)
        pack = [
            others[start % len(others)],
            others[(start + 5) % len(others)],
            others[(start + 10) % len(others)],
        ]
        deduped = []
        for item_id in pack:
            if item_id not in deduped:
                deduped.append(item_id)
        cursor = 0
        while len(deduped) < 3:
            candidate = others[(start + cursor) % len(others)]
            if candidate not in deduped:
                deduped.append(candidate)
            cursor += 1
        packs.append(tuple(deduped))
    return packs


def run_doer_sequence(
    doer,
    doer_params,
    local_obs,
    proprioception,
    message_sequence,
    menu_images,
    hidden_size,
    pick_only_phase,
    pick_available_sequence=None,
):
    carry = doer.initialize_carry(batch_size=1, hidden_size=hidden_size)
    action_trace = []
    logits_trace = []

    for step_idx, word in enumerate(message_sequence):
        message = jnp.asarray([word], dtype=jnp.float32)
        carry, logits = doer.apply(
            {"params": doer_params},
            carry,
            local_obs,
            proprioception,
            message,
            menu_images,
        )
        pick_available = None
        if pick_available_sequence is not None:
            pick_available = jnp.asarray([pick_available_sequence[step_idx]], dtype=bool)
        masked_logits = mask_pick_actions_until_menu_visible(
            logits,
            menu_images,
            pick_only_phase=pick_only_phase,
            pick_available=pick_available,
        )
        action = int(jnp.argmax(masked_logits[0]))
        action_trace.append(action)
        logits_trace.append(np.asarray(masked_logits[0], dtype=np.float32).tolist())

    return {
        "action_trace": action_trace,
        "action_trace_labels": [action_label(action) for action in action_trace],
        "final_action": action_trace[-1],
        "first_pick_step": next(
            (step_idx for step_idx, action in enumerate(action_trace) if action >= 5),
            None,
        ),
        "masked_logits_trace": logits_trace,
    }


def build_eval_contexts(args):
    nav_env = build_env(args, selection_phase_level=2)
    fixed_positions = jnp.asarray(
        [[1, nav_env._left_col], [1, nav_env._right_col]],
        dtype=jnp.int32,
    )
    nav_obs, _ = nav_env.reset(jax.random.PRNGKey(0), fixed_positions=fixed_positions)

    pick_env = build_env(args, selection_phase_level=1)
    pick_obs, _ = pick_env.reset(jax.random.PRNGKey(1), fixed_positions=fixed_positions)
    return nav_env, nav_obs, pick_env, pick_obs


def probe_navigation_semantics(doer, doer_params, nav_obs, fsq_levels, sequence_length, hidden_size):
    results = {"doer_a": {}, "doer_b": {}}
    zero_menu = jnp.zeros((1, 4, 5, 5, 3), dtype=jnp.float32)

    for doer_idx, doer_key in enumerate(DOER_KEYS):
        local_obs = nav_obs["local_views"][doer_idx][None, ...]
        proprioception = nav_obs["proprioceptions"][doer_idx][None, ...]
        for message_sequence in all_message_sequences(fsq_levels, sequence_length):
            rollout = run_doer_sequence(
                doer,
                doer_params,
                local_obs,
                proprioception,
                message_sequence,
                zero_menu,
                hidden_size=hidden_size,
                pick_only_phase=False,
                pick_available_sequence=[False] * sequence_length,
            )
            results[doer_key][message_sequence_key(message_sequence)] = {
                "action_trace": rollout["action_trace_labels"],
                "final_action": action_label(rollout["final_action"]),
            }
    return results


def probe_selection_semantics(
    doer,
    doer_params,
    pick_env,
    pick_obs,
    fsq_levels,
    distractor_packs,
    sequence_length,
    hidden_size,
):
    bank = pick_env.item_bank
    num_items = int(bank.shape[0])
    results = {"doer_a": {}, "doer_b": {}}
    pick_available_sequence = [
        step_idx >= pick_env.pick_object_listen_steps for step_idx in range(sequence_length)
    ]

    for doer_idx, doer_key in enumerate(DOER_KEYS):
        local_obs = pick_obs["local_views"][doer_idx][None, ...]
        proprioception = pick_obs["proprioceptions"][doer_idx][None, ...]

        for message_sequence in all_message_sequences(fsq_levels, sequence_length):
            per_target = {}

            for target_id in range(num_items):
                first_pick_target_hits = 0
                final_pick_target_hits = 0
                total_trials = 0
                first_pick_objects = []
                final_pick_objects = []
                no_pick = 0

                for distractors in distractor_packs_for_target(
                    target_id,
                    num_items=num_items,
                    num_packs=distractor_packs,
                ):
                    base_menu = list(distractors)
                    for slot_idx in range(4):
                        menu_ids = base_menu.copy()
                        menu_ids.insert(slot_idx, target_id)
                        menu_images = bank[jnp.asarray(menu_ids, dtype=jnp.int32)][None, ...]
                        rollout = run_doer_sequence(
                            doer,
                            doer_params,
                            local_obs,
                            proprioception,
                            message_sequence,
                            menu_images,
                            hidden_size=hidden_size,
                            pick_only_phase=True,
                            pick_available_sequence=pick_available_sequence,
                        )
                        total_trials += 1
                        first_pick_step = rollout["first_pick_step"]
                        final_action = rollout["final_action"]

                        if first_pick_step is None:
                            no_pick += 1
                            first_pick_objects.append(None)
                        else:
                            first_pick_action = rollout["action_trace"][first_pick_step]
                            first_pick_item = int(menu_ids[first_pick_action - 5])
                            first_pick_objects.append(first_pick_item)
                            if first_pick_item == target_id:
                                first_pick_target_hits += 1

                        if final_action < 5:
                            final_pick_objects.append(None)
                        else:
                            final_pick_item = int(menu_ids[final_action - 5])
                            final_pick_objects.append(final_pick_item)
                            if final_pick_item == target_id:
                                final_pick_target_hits += 1

                def dominant_choice(chosen_items):
                    object_counts = {}
                    for chosen_item in chosen_items:
                        if chosen_item is None:
                            continue
                        object_counts[chosen_item] = object_counts.get(chosen_item, 0) + 1
                    dominant_choice_id = None
                    dominant_choice_count = 0
                    for choice_id, count in object_counts.items():
                        if count > dominant_choice_count:
                            dominant_choice_id = choice_id
                            dominant_choice_count = count
                    return dominant_choice_id, dominant_choice_count

                first_dominant_choice_id, first_dominant_choice_count = dominant_choice(first_pick_objects)
                final_dominant_choice_id, final_dominant_choice_count = dominant_choice(final_pick_objects)

                per_target[item_label(target_id)] = {
                    "item_id": target_id,
                    "first_pick_target_hit_rate": (
                        first_pick_target_hits / total_trials if total_trials else 0.0
                    ),
                    "final_pick_target_hit_rate": (
                        final_pick_target_hits / total_trials if total_trials else 0.0
                    ),
                    "no_pick_rate": no_pick / total_trials if total_trials else 0.0,
                    "dominant_first_pick_object": (
                        item_label(first_dominant_choice_id)
                        if first_dominant_choice_id is not None
                        else None
                    ),
                    "dominant_first_pick_rate": (
                        first_dominant_choice_count / total_trials if total_trials else 0.0
                    ),
                    "dominant_final_pick_object": (
                        item_label(final_dominant_choice_id)
                        if final_dominant_choice_id is not None
                        else None
                    ),
                    "dominant_final_pick_rate": (
                        final_dominant_choice_count / total_trials if total_trials else 0.0
                    ),
                }

            ranked_targets = sorted(
                per_target.values(),
                key=lambda entry: (
                    entry["first_pick_target_hit_rate"],
                    entry["final_pick_target_hit_rate"],
                    entry["dominant_final_pick_rate"],
                ),
                reverse=True,
            )
            results[doer_key][message_sequence_key(message_sequence)] = {
                "best_target_label": item_label(ranked_targets[0]["item_id"]),
                "best_first_pick_hit_rate": ranked_targets[0]["first_pick_target_hit_rate"],
                "best_final_pick_hit_rate": ranked_targets[0]["final_pick_target_hit_rate"],
                "best_matching_first_pick_object": ranked_targets[0]["dominant_first_pick_object"],
                "best_matching_final_pick_object": ranked_targets[0]["dominant_final_pick_object"],
                "top_candidates": ranked_targets[:3],
                "all_targets": per_target,
            }

    return results


def build_counterfactual_summary(message_sequences, navigation_results, selection_results):
    lines = []
    lines.append(
        "Because the doer is recurrent, message-pair probes are more faithful than single-word probes."
    )
    lines.append(
        "In selection, first_pick_* is the most meaningful field because the environment reacts to the first valid pick."
    )
    for message_sequence in message_sequences:
        seq_key = message_sequence_key(message_sequence)
        nav_a = navigation_results["doer_a"][seq_key]
        nav_b = navigation_results["doer_b"][seq_key]
        sel_a = selection_results["doer_a"][seq_key]
        sel_b = selection_results["doer_b"][seq_key]
        lines.append(
            " | ".join(
                [
                    f"msg_seq={seq_key}",
                    f"nav_a={'->'.join(nav_a['action_trace'])}",
                    f"nav_b={'->'.join(nav_b['action_trace'])}",
                    (
                        f"pick_a={sel_a['best_target_label']} "
                        f"(first={sel_a['best_first_pick_hit_rate']:.2f})"
                    ),
                    (
                        f"pick_b={sel_b['best_target_label']} "
                        f"(first={sel_b['best_first_pick_hit_rate']:.2f})"
                    ),
                ]
            )
        )
    return lines


def context_label(step_record, agent_idx):
    if step_record["has_selected"][agent_idx]:
        return "locked_after_pick"
    if step_record["actions"][agent_idx] >= 5:
        return "pick"
    if step_record["pick_available"][agent_idx]:
        return "menu_visible"
    if step_record["at_goal"][agent_idx]:
        return "at_goal_wait"
    return "navigate"


def sorted_distribution(counter):
    total = sum(counter.values())
    return [
        {"label": label, "count": count, "rate": count / total if total else 0.0}
        for label, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def sample_context(step_record, agent_idx):
    return {
        "step": int(step_record["step"]),
        "position": step_record["positions"][agent_idx],
        "goal": step_record["goals"][agent_idx],
        "target_label": step_record["target_labels"][agent_idx],
        "target_slot": step_record["target_slots"][agent_idx],
        "menu_labels": step_record["menu_labels"][agent_idx],
        "message": step_record["messages"][agent_idx],
        "action": step_record["action_labels"][agent_idx],
        "pick_available": bool(step_record["pick_available"][agent_idx]),
        "at_goal": bool(step_record["at_goal"][agent_idx]),
        "chosen_item_label": step_record["chosen_item_labels"][agent_idx],
    }


def summarize_empirical_bucket(bucket):
    total_picks = sum(count for label, count in bucket["next_action_counts"].items() if label.startswith("pick_"))
    return {
        "count": bucket["count"],
        "dominant_next_action": max(
            bucket["next_action_counts"].items(),
            key=lambda item: (item[1], item[0]),
        )[0],
        "next_action_distribution": sorted_distribution(bucket["next_action_counts"]),
        "current_action_distribution": sorted_distribution(bucket["current_action_counts"]),
        "context_distribution": sorted_distribution(bucket["context_counts"]),
        "target_distribution": sorted_distribution(bucket["target_counts"]),
        "target_slot_distribution": sorted_distribution(bucket["target_slot_counts"]),
        "pick_target_hit_rate_among_picks": (
            bucket["pick_target_hits"] / total_picks if total_picks else 0.0
        ),
        "samples": bucket["samples"],
    }


def analyze_rollout_messages(episodes, sample_limit=4):
    analysis = {}

    for agent_idx, doer_key in enumerate(DOER_KEYS):
        single_buckets = {}
        pair_buckets = {}

        for episode_idx, episode in enumerate(episodes):
            steps = episode["steps"]
            for step_idx, step_record in enumerate(steps):
                single_key = message_key(step_record["messages"][agent_idx])
                if single_key not in single_buckets:
                    single_buckets[single_key] = {
                        "count": 0,
                        "current_action_counts": Counter(),
                        "next_action_counts": Counter(),
                        "context_counts": Counter(),
                        "target_counts": Counter(),
                        "target_slot_counts": Counter(),
                        "pick_target_hits": 0,
                        "samples": [],
                    }
                single_bucket = single_buckets[single_key]
                single_bucket["count"] += 1
                single_bucket["current_action_counts"][step_record["action_labels"][agent_idx]] += 1
                single_bucket["next_action_counts"][step_record["action_labels"][agent_idx]] += 1
                single_bucket["context_counts"][context_label(step_record, agent_idx)] += 1
                single_bucket["target_counts"][step_record["target_labels"][agent_idx]] += 1
                single_bucket["target_slot_counts"][str(step_record["target_slots"][agent_idx])] += 1
                chosen_item = step_record["chosen_items"][agent_idx]
                if chosen_item is not None and chosen_item == step_record["target_items"][agent_idx]:
                    single_bucket["pick_target_hits"] += 1
                if len(single_bucket["samples"]) < sample_limit:
                    single_bucket["samples"].append(
                        {
                            "episode": episode_idx,
                            **sample_context(step_record, agent_idx),
                        }
                    )

                if step_idx + 1 >= len(steps):
                    continue

                next_step = steps[step_idx + 1]
                pair_key = (
                    f"{message_key(step_record['messages'][agent_idx])} -> "
                    f"{message_key(next_step['messages'][agent_idx])}"
                )
                if pair_key not in pair_buckets:
                    pair_buckets[pair_key] = {
                        "count": 0,
                        "current_action_counts": Counter(),
                        "next_action_counts": Counter(),
                        "context_counts": Counter(),
                        "target_counts": Counter(),
                        "target_slot_counts": Counter(),
                        "pick_target_hits": 0,
                        "samples": [],
                    }
                pair_bucket = pair_buckets[pair_key]
                pair_bucket["count"] += 1
                pair_bucket["current_action_counts"][step_record["action_labels"][agent_idx]] += 1
                pair_bucket["next_action_counts"][next_step["action_labels"][agent_idx]] += 1
                pair_bucket["context_counts"][context_label(next_step, agent_idx)] += 1
                pair_bucket["target_counts"][next_step["target_labels"][agent_idx]] += 1
                pair_bucket["target_slot_counts"][str(next_step["target_slots"][agent_idx])] += 1
                next_chosen_item = next_step["chosen_items"][agent_idx]
                if next_chosen_item is not None and next_chosen_item == next_step["target_items"][agent_idx]:
                    pair_bucket["pick_target_hits"] += 1
                if len(pair_bucket["samples"]) < sample_limit:
                    pair_bucket["samples"].append(
                        {
                            "episode": episode_idx,
                            "prev_step": sample_context(step_record, agent_idx),
                            "next_step": sample_context(next_step, agent_idx),
                        }
                    )

        analysis[doer_key] = {
            "single_message_summary": {
                key: summarize_empirical_bucket(bucket)
                for key, bucket in sorted(
                    single_buckets.items(),
                    key=lambda item: (-item[1]["count"], item[0]),
                )
            },
            "message_pair_summary": {
                key: summarize_empirical_bucket(bucket)
                for key, bucket in sorted(
                    pair_buckets.items(),
                    key=lambda item: (-item[1]["count"], item[0]),
                )
            },
        }

    return analysis


def build_empirical_summary_lines(empirical_analysis, limit):
    lines = []
    for doer_key in DOER_KEYS:
        lines.append(f"{doer_key}:")
        pair_items = list(empirical_analysis[doer_key]["message_pair_summary"].items())[:limit]
        for pair_key, summary in pair_items:
            lines.append(
                " | ".join(
                    [
                        f"pair={pair_key}",
                        f"count={summary['count']}",
                        f"dominant_next={summary['dominant_next_action']}",
                        f"pick_hit@picks={summary['pick_target_hit_rate_among_picks']:.2f}",
                    ]
                )
            )
    return lines


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    gifs_dir = output_dir / "gifs"
    output_dir.mkdir(parents=True, exist_ok=True)
    gifs_dir.mkdir(parents=True, exist_ok=True)

    fsq_levels = parse_fsq_levels(args.fsq_levels)
    params, resolved_checkpoint = restore_params(args.checkpoint)

    expected_message_dim = infer_message_dim_from_params(params)
    if len(fsq_levels) != expected_message_dim:
        raise ValueError(
            "Checkpoint/message mismatch: the restored doer expects "
            f"{expected_message_dim} message dimensions, but --fsq-levels {list(fsq_levels)} "
            f"has {len(fsq_levels)} entries."
        )

    env = build_env(args)
    seer = Seer(
        fsq_levels=fsq_levels,
        num_actions=env.num_actions,
        num_message_heads=env.num_doers,
    )
    doer = Doer(fsq_levels=fsq_levels, num_actions=env.num_actions)

    rng = jax.random.PRNGKey(args.seed)
    episodes = []
    for episode_idx in range(args.num_episodes):
        gif_path = None
        if episode_idx < args.num_visualizations:
            gif_path = gifs_dir / f"greedy_episode_{episode_idx:03d}.gif"
        rng, episode_record = run_policy_episode(
            env,
            params,
            rng,
            seer,
            doer,
            hidden_size=args.hidden_size,
            max_steps=env.phase_max_steps,
            gif_path=gif_path,
        )
        episode_record["episode_index"] = episode_idx
        episodes.append(episode_record)

    traces_path = output_dir / "greedy_rollout_traces.json"
    traces_path.write_text(json.dumps(episodes, indent=2))

    empirical_analysis = analyze_rollout_messages(episodes)
    empirical_summary_lines = build_empirical_summary_lines(
        empirical_analysis,
        limit=args.print_summary_limit,
    )

    nav_env, nav_obs, pick_env, pick_obs = build_eval_contexts(args)
    message_sequences = all_message_sequences(fsq_levels, args.sequence_length)
    navigation_results = probe_navigation_semantics(
        doer,
        params["doer"],
        nav_obs,
        fsq_levels,
        args.sequence_length,
        hidden_size=args.hidden_size,
    )
    selection_results = probe_selection_semantics(
        doer,
        params["doer"],
        pick_env,
        pick_obs,
        fsq_levels,
        distractor_packs=args.distractor_packs,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
    )
    counterfactual_summary_lines = build_counterfactual_summary(
        message_sequences,
        navigation_results,
        selection_results,
    )

    success_rate = sum(1 for episode in episodes if episode["success"]) / max(len(episodes), 1)
    report = {
        "checkpoint_requested": args.checkpoint,
        "checkpoint_resolved": str(resolved_checkpoint),
        "output_dir": str(output_dir.resolve()),
        "fsq_levels": list(fsq_levels),
        "num_messages": int(np.prod(np.asarray(fsq_levels, dtype=np.int32))),
        "sequence_length": args.sequence_length,
        "num_message_sequences": len(message_sequences),
        "environment": {
            "selection_phase_level": int(args.selection_phase_level),
            "doer_perception_level": int(args.doer_perception_level),
            "grid_height": int(args.grid_height),
            "grid_width": int(args.grid_width),
            "local_view_size": int(args.local_view_size),
            "corridor_length": int(args.corridor_length),
            "max_steps": int(args.max_steps),
            "pick_object_max_steps": int(args.pick_object_max_steps),
            "pick_object_listen_steps": int(args.pick_object_listen_steps),
        },
        "analysis_note": (
            "Single-message lookups are incomplete because the doer is recurrent. "
            "This report therefore includes both controlled sequence probes and empirical "
            "message-pair statistics from real greedy episodes."
        ),
        "greedy_eval": {
            "num_episodes": len(episodes),
            "success_rate": success_rate,
            "visualization_paths": [
                episode["gif_path"] for episode in episodes if "gif_path" in episode
            ],
            "traces_path": str(traces_path),
            "episode_summaries": [
                {
                    "episode_index": episode["episode_index"],
                    "success": episode["success"],
                    "num_steps": episode["num_steps"],
                    "gif_path": episode.get("gif_path"),
                }
                for episode in episodes
            ],
        },
        "empirical_rollout_analysis": empirical_analysis,
        "empirical_summary_lines": empirical_summary_lines,
        "counterfactual_navigation_results": navigation_results,
        "counterfactual_selection_results": selection_results,
        "counterfactual_summary_lines": counterfactual_summary_lines,
        "item_labels": {str(item_id): item_label(item_id) for item_id in range(16)},
    }

    report_path = output_dir / "communication_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"Resolved checkpoint: {resolved_checkpoint}")
    print(f"Greedy rollout success rate: {success_rate:.3f} over {len(episodes)} episodes")
    print(f"Saved greedy traces to {traces_path}")
    print(f"Saved communication report to {report_path}")
    print("")
    print("Top Empirical Message-Pair Summaries")
    print("=" * 72)
    for line in empirical_summary_lines:
        print(line)


if __name__ == "__main__":
    main()
