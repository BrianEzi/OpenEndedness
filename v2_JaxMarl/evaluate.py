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


def item_label(item_id: int) -> str:
    color = COLOR_NAMES[item_id // 4]
    shape = SHAPE_NAMES[item_id % 4]
    return f"{color}_{shape}"


def all_messages(fsq_levels):
    return [tuple(word) for word in itertools.product(*[range(level) for level in fsq_levels])]


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


def run_doer(doer, doer_params, carry, local_obs, proprioception, message, menu_images, pick_only_phase):
    _, logits = doer.apply(
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
    return action


def probe_navigation_semantics(doer, doer_params, nav_obs, fsq_levels):
    carry = doer.initialize_carry(batch_size=1, hidden_size=128)
    results = {"doer_a": {}, "doer_b": {}}
    zero_menu = jnp.zeros((1, 4, 5, 5, 3), dtype=jnp.float32)

    for doer_idx, doer_key in enumerate(("doer_a", "doer_b")):
        local_obs = nav_obs["local_views"][doer_idx][None, ...]
        proprioception = nav_obs["proprioceptions"][doer_idx][None, ...]
        for word in all_messages(fsq_levels):
            message = jnp.asarray([word], dtype=jnp.float32)
            action = run_doer(
                doer,
                doer_params,
                carry,
                local_obs,
                proprioception,
                message,
                zero_menu,
                pick_only_phase=False,
            )
            results[doer_key][str(word)] = NAV_ACTION_LABELS[action]
    return results


def probe_selection_semantics(doer, doer_params, pick_env, pick_obs, fsq_levels, distractor_packs):
    carry = doer.initialize_carry(batch_size=1, hidden_size=128)
    bank = pick_env.item_bank
    num_items = int(bank.shape[0])
    results = {"doer_a": {}, "doer_b": {}}

    for doer_idx, doer_key in enumerate(("doer_a", "doer_b")):
        local_obs = pick_obs["local_views"][doer_idx][None, ...]
        proprioception = pick_obs["proprioceptions"][doer_idx][None, ...]

        for word in all_messages(fsq_levels):
            message = jnp.asarray([word], dtype=jnp.float32)
            per_target = {}

            for target_id in range(num_items):
                target_hits = 0
                total_trials = 0
                chosen_objects = []
                stayed = 0
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
                        action = run_doer(
                            doer,
                            doer_params,
                            carry,
                            local_obs,
                            proprioception,
                            message,
                            menu_images,
                            pick_only_phase=True,
                        )
                        total_trials += 1
                        if action < 5:
                            stayed += 1
                            chosen_objects.append(None)
                            continue
                        chosen_item = int(menu_ids[action - 5])
                        chosen_objects.append(chosen_item)
                        if chosen_item == target_id:
                            target_hits += 1

                object_counts = {}
                for chosen_item in chosen_objects:
                    if chosen_item is None:
                        continue
                    object_counts[chosen_item] = object_counts.get(chosen_item, 0) + 1

                dominant_choice_id = None
                dominant_choice_count = 0
                for choice_id, count in object_counts.items():
                    if count > dominant_choice_count:
                        dominant_choice_id = choice_id
                        dominant_choice_count = count

                per_target[item_label(target_id)] = {
                    "item_id": target_id,
                    "target_hit_rate": target_hits / total_trials if total_trials else 0.0,
                    "stay_rate": stayed / total_trials if total_trials else 0.0,
                    "dominant_chosen_object": (
                        item_label(dominant_choice_id) if dominant_choice_id is not None else None
                    ),
                    "dominant_choice_rate": dominant_choice_count / total_trials if total_trials else 0.0,
                }

            ranked_targets = sorted(
                per_target.values(),
                key=lambda entry: (entry["target_hit_rate"], entry["dominant_choice_rate"]),
                reverse=True,
            )
            results[doer_key][str(word)] = {
                "best_matching_object": ranked_targets[0]["dominant_chosen_object"],
                "best_target_label": item_label(ranked_targets[0]["item_id"]),
                "best_target_hit_rate": ranked_targets[0]["target_hit_rate"],
                "top_candidates": ranked_targets[:3],
                "all_targets": per_target,
            }

    return results


def build_summary(messages, navigation_results, selection_results):
    lines = []
    lines.append("Navigation actions are limited to: stay, up, right, down, left. There are no turning actions in this env.")
    for word in messages:
        word_key = str(word)
        nav_a = navigation_results["doer_a"][word_key]
        nav_b = navigation_results["doer_b"][word_key]
        sel_a = selection_results["doer_a"][word_key]
        sel_b = selection_results["doer_b"][word_key]
        lines.append(
            " | ".join(
                [
                    f"msg={word_key}",
                    f"nav_a={nav_a}",
                    f"nav_b={nav_b}",
                    f"obj_a={sel_a['best_target_label']} ({sel_a['best_target_hit_rate']:.2f})",
                    f"obj_b={sel_b['best_target_label']} ({sel_b['best_target_hit_rate']:.2f})",
                ]
            )
        )
    return lines


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fsq_levels = parse_fsq_levels(args.fsq_levels)
    params = restore_params(checkpoint_path)
    doer = Doer(fsq_levels=fsq_levels, num_actions=9)

    nav_env, nav_obs, pick_env, pick_obs = build_eval_contexts()
    navigation_results = probe_navigation_semantics(
        doer,
        params["doer"],
        nav_obs,
        fsq_levels,
    )
    selection_results = probe_selection_semantics(
        doer,
        params["doer"],
        pick_env,
        pick_obs,
        fsq_levels,
        distractor_packs=args.distractor_packs,
    )

    messages = all_messages(fsq_levels)
    summary_lines = build_summary(messages, navigation_results, selection_results)

    report = {
        "checkpoint": str(checkpoint_path),
        "fsq_levels": list(fsq_levels),
        "num_messages": int(np.prod(np.asarray(fsq_levels, dtype=np.int32))),
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
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
