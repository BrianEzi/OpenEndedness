import argparse
import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

from envs.two_doer_grid import TwoDoerBottleneckEnv
from models.doer import Doer
from training.action_masking import mask_pick_actions_until_menu_visible


NAV_ACTION_LABELS = ("stay", "up", "right", "down", "left")
COLOR_NAMES = ("red", "green", "blue", "yellow")
SHAPE_NAMES = ("solid_square", "plus", "x", "frame")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Probe a trained two-doer checkpoint by enumerating all discrete messages "
            "and measuring the doer actions they induce."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints-extended-comm/checkpoint_19999744",
        help="Path to a checkpoint directory or its parent directory.",
    )
    parser.add_argument(
        "--fsq-levels",
        type=str,
        default="4,4",
        help="Comma-separated FSQ levels used by the checkpoint, e.g. 4,4 or 2,2,2,2.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/eval/communication_protocol.json",
        help="Where to save the detailed JSON report.",
    )
    parser.add_argument(
        "--distractor-packs",
        type=int,
        default=4,
        help="How many distractor menu variants to test per target object.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=1,
        choices=(1, 2, 3),
        help="How many consecutive messages to feed through the doer's LSTM before reading out behavior.",
    )
    parser.add_argument(
        "--print-summary-limit",
        type=int,
        default=64,
        help="Maximum number of summary lines to print to stdout. Full results always go to JSON.",
    )
    return parser.parse_args()


def parse_fsq_levels(levels_str: str):
    return tuple(int(level.strip()) for level in levels_str.split(",") if level.strip())


def resolve_checkpoint_location(checkpoint_path: Path):
    if checkpoint_path.name.startswith("checkpoint_"):
        step = int(checkpoint_path.name.split("_")[-1])
        return checkpoint_path.parent, step
    return checkpoint_path, None


def restore_params(checkpoint_path: Path):
    restore_dir, step = resolve_checkpoint_location(checkpoint_path)
    params = checkpoints.restore_checkpoint(
        ckpt_dir=str(restore_dir),
        target=None,
        step=step,
        prefix="checkpoint_",
    )
    if "doer" not in params:
        raise KeyError("Checkpoint does not contain `doer` parameters.")
    return params


def infer_message_dim_from_params(params) -> int:
    try:
        return int(params["doer"]["Dense_0"]["kernel"].shape[0])
    except KeyError as exc:
        raise KeyError(
            "Could not infer message dimension from checkpoint params at "
            "params['doer']['Dense_0']['kernel']."
        ) from exc


def item_label(item_id: int) -> str:
    color = COLOR_NAMES[item_id // 4]
    shape = SHAPE_NAMES[item_id % 4]
    return f"{color}_{shape}"


def all_messages(fsq_levels):
    return [tuple(word) for word in itertools.product(*[range(level) for level in fsq_levels])]


def all_message_sequences(fsq_levels, sequence_length: int):
    messages = all_messages(fsq_levels)
    return [tuple(seq) for seq in itertools.product(messages, repeat=sequence_length)]


def message_sequence_key(message_sequence) -> str:
    return " -> ".join(str(word) for word in message_sequence)


def build_eval_contexts():
    nav_env = TwoDoerBottleneckEnv(selection_phase_level=2)
    fixed_positions = jnp.asarray(
        [[1, nav_env._left_col], [1, nav_env._right_col]],
        dtype=jnp.int32,
    )
    nav_obs, _ = nav_env.reset(jax.random.PRNGKey(0), fixed_positions=fixed_positions)

    pick_env = TwoDoerBottleneckEnv(selection_phase_level=1)
    pick_obs, _ = pick_env.reset(jax.random.PRNGKey(1), fixed_positions=fixed_positions)

    return nav_env, nav_obs, pick_env, pick_obs


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


def action_label(action: int) -> str:
    if action < 5:
        return NAV_ACTION_LABELS[action]
    return f"pick_{action - 5}"


def run_doer_sequence(
    doer,
    doer_params,
    local_obs,
    proprioception,
    message_sequence,
    menu_images,
    pick_only_phase,
):
    carry = doer.initialize_carry(batch_size=1, hidden_size=128)
    action_trace = []
    logits_trace = []

    for word in message_sequence:
        message = jnp.asarray([word], dtype=jnp.float32)
        carry, logits = doer.apply(
            {"params": doer_params},
            carry,
            local_obs,
            proprioception,
            message,
            menu_images,
        )
        masked_logits = mask_pick_actions_until_menu_visible(
            logits,
            menu_images,
            pick_only_phase=pick_only_phase,
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


def probe_navigation_semantics(doer, doer_params, nav_obs, fsq_levels, sequence_length):
    results = {"doer_a": {}, "doer_b": {}}
    zero_menu = jnp.zeros((1, 4, 5, 5, 3), dtype=jnp.float32)

    for doer_idx, doer_key in enumerate(("doer_a", "doer_b")):
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
                pick_only_phase=False,
            )
            results[doer_key][message_sequence_key(message_sequence)] = {
                "action_trace": rollout["action_trace_labels"],
                "final_action": NAV_ACTION_LABELS[rollout["final_action"]],
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
):
    bank = pick_env.item_bank
    num_items = int(bank.shape[0])
    results = {"doer_a": {}, "doer_b": {}}

    for doer_idx, doer_key in enumerate(("doer_a", "doer_b")):
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
                            pick_only_phase=True,
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


def build_summary(message_sequences, navigation_results, selection_results):
    lines = []
    lines.append(
        "Navigation actions are limited to: stay, up, right, down, left. There are no turning actions in this env."
    )
    lines.append(
        "For selection, first_pick_* is the behaviorally meaningful metric because the environment would react to the first pick in the sequence."
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
                        f"obj_a={sel_a['best_target_label']} "
                        f"(first={sel_a['best_first_pick_hit_rate']:.2f}, final={sel_a['best_final_pick_hit_rate']:.2f})"
                    ),
                    (
                        f"obj_b={sel_b['best_target_label']} "
                        f"(first={sel_b['best_first_pick_hit_rate']:.2f}, final={sel_b['best_final_pick_hit_rate']:.2f})"
                    ),
                ]
            )
        )
    return lines


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fsq_levels = parse_fsq_levels(args.fsq_levels)
    params = restore_params(checkpoint_path)
    expected_message_dim = infer_message_dim_from_params(params)
    if len(fsq_levels) != expected_message_dim:
        raise ValueError(
            "Checkpoint/message mismatch: the restored doer expects "
            f"{expected_message_dim} message dimensions, but --fsq-levels {list(fsq_levels)} "
            f"has {len(fsq_levels)}. Use an fsq-level list with {expected_message_dim} entries."
        )
    doer = Doer(fsq_levels=fsq_levels, num_actions=9)
    message_sequences = all_message_sequences(fsq_levels, args.sequence_length)

    nav_env, nav_obs, pick_env, pick_obs = build_eval_contexts()
    navigation_results = probe_navigation_semantics(
        doer,
        params["doer"],
        nav_obs,
        fsq_levels,
        args.sequence_length,
    )
    selection_results = probe_selection_semantics(
        doer,
        params["doer"],
        pick_env,
        pick_obs,
        fsq_levels,
        distractor_packs=args.distractor_packs,
        sequence_length=args.sequence_length,
    )

    summary_lines = build_summary(message_sequences, navigation_results, selection_results)

    report = {
        "checkpoint": str(checkpoint_path),
        "fsq_levels": list(fsq_levels),
        "num_messages": int(np.prod(np.asarray(fsq_levels, dtype=np.int32))),
        "sequence_length": args.sequence_length,
        "num_message_sequences": len(message_sequences),
        "item_labels": {str(item_id): item_label(item_id) for item_id in range(16)},
        "navigation_note": "No turning actions exist in the two-doer bottleneck env; navigation semantics are stay/up/right/down/left only.",
        "navigation_results": navigation_results,
        "selection_results": selection_results,
        "summary_lines": summary_lines,
    }

    output_path.write_text(json.dumps(report, indent=2))
    print(f"Saved communication evaluation to {output_path}")
    print("")
    print("Protocol Summary")
    print("=" * 72)
    for line in summary_lines[: args.print_summary_limit]:
        print(line)
    if len(summary_lines) > args.print_summary_limit:
        print(
            f"... truncated {len(summary_lines) - args.print_summary_limit} additional lines. "
            f"See {output_path} for the full report."
        )


if __name__ == "__main__":
    main()
