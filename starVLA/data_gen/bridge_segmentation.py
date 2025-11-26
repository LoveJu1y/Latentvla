"""
Offline trajectory segmentation for LeRobot-formatted BRIDGE-Orig dataset.

This script does NOT modify the main training code. It reads episodes from
`/share/project/baishuanghao/data/bridge_orig_lerobot`, performs rule-based
subtask segmentation using only state/action, and writes a JSONL file with
per-episode segmentation metadata.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

try:
    # Prefer pyarrow for parquet reading to avoid adding heavy dependencies
    import pyarrow.parquet as pq  # type: ignore
except ImportError as e:  # pragma: no cover - simple import guard
    raise ImportError(
        "pyarrow is required to run bridge_segmentation.py. "
        "Please install it in your environment, e.g. `pip install pyarrow`."
    ) from e


BRIDGE_ROOT_DEFAULT = "/share/project/baishuanghao/data/bridge_orig_lerobot"


SEGMENT_TYPES = [
    "move_to_object",
    "grasp_object",
    "move_to_goal",
    "place_object",
    "retract",
]


@dataclass
class Segment:
    segment_id: int
    segment_type: str
    start_t: int
    end_t: int


@dataclass
class Cycle:
    cycle_id: int
    segments: List[Segment]


def load_episode_parquet(root: Path, episode_index: int, chunk_size: int) -> Dict[str, np.ndarray]:
    """
    Load a single episode parquet file into numpy arrays.

    We avoid pandas and instead rely on pyarrow directly.
    """
    chunk_index = episode_index // chunk_size
    parquet_path = root / "data" / f"chunk-{chunk_index:03d}" / f"episode_{episode_index:06d}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")

    table = pq.read_table(parquet_path)
    cols = {name: table[name].to_numpy() for name in table.column_names}

    # We expect LeRobot-style packed arrays for state and action
    # observation.state: shape (T, 8)
    # action: shape (T, 7)
    state = np.stack(cols["observation.state"])
    action = np.stack(cols["action"])
    # gripper: last dim
    return {
        "state": state,
        "action": action,
    }


def derive_features(state: np.ndarray, action: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Derive position, velocity, height and gripper features from low-dim state/action.

    state: [T, 8] -> xyz, rpy, pad, gripper
    action: [T, 7] -> xyz, rpy, gripper
    """
    assert state.ndim == 2 and state.shape[1] >= 8, f"Unexpected state shape: {state.shape}"
    assert action.ndim == 2 and action.shape[1] >= 7, f"Unexpected action shape: {action.shape}"

    pos = state[:, 0:3]  # x, y, z
    z = state[:, 2]
    # simple position-based velocity
    vel = np.linalg.norm(np.diff(pos, axis=0, prepend=pos[[0]]), axis=-1)

    # choose gripper from action if available, else from state
    g_action = action[:, -1]
    g_state = state[:, -1]
    # heuristic: if action gripper has non-trivial variance, prefer it
    if np.std(g_action) > 1e-4:
        g_raw = g_action
    else:
        g_raw = g_state

    return {
        "pos": pos,
        "z": z,
        "vel": vel,
        "gripper_raw": g_raw,
    }


def smooth_signal(x: np.ndarray, k: int = 3) -> np.ndarray:
    """Simple moving-average smoothing with window size k."""
    if k <= 1:
        return x
    pad = k // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / k
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(x.dtype)


def binarize_gripper(g_raw: np.ndarray, th_open: float = 0.3, th_close: float = 0.7) -> np.ndarray:
    """
    Convert continuous gripper signal to binary open/close.
    """
    g_smooth = smooth_signal(g_raw, k=5)
    g_bin = np.zeros_like(g_smooth, dtype=np.int64)
    # we assume raw in [0,1], but clamp to be safe
    g_norm = (g_smooth - g_smooth.min()) / (g_smooth.ptp() + 1e-6)
    g_bin[g_norm >= th_close] = 1
    g_bin[g_norm <= th_open] = 0
    # between open/close thresholds: keep previous state
    for t in range(1, len(g_bin)):
        if th_open < g_norm[t] < th_close:
            g_bin[t] = g_bin[t - 1]
    return g_bin


def detect_gripper_events(g_bin: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Detect open->close (grasp) and close->open (place) events.

    Returns:
        t_close_list, t_open_list
    """
    t_close_list: List[int] = []
    t_open_list: List[int] = []
    for t in range(1, len(g_bin)):
        if g_bin[t - 1] == 0 and g_bin[t] == 1:
            t_close_list.append(t)
        elif g_bin[t - 1] == 1 and g_bin[t] == 0:
            t_open_list.append(t)
    return t_close_list, t_open_list


def refine_center_by_velocity_and_height(
    vel: np.ndarray,
    z: np.ndarray,
    start: int,
    end: int,
    z_weight: float = 0.5,
) -> int:
    """
    Heuristic: within [start, end], choose a "center" index that has
    relatively small velocity and relatively low height (closer to table).
    """
    start = max(0, start)
    end = min(len(vel) - 1, end)
    if end <= start:
        return start

    seg_vel = vel[start : end + 1]
    seg_z = z[start : end + 1]

    v_norm = (seg_vel - seg_vel.min()) / (seg_vel.ptp() + 1e-6)
    z_norm = (seg_z - seg_z.min()) / (seg_z.ptp() + 1e-6)

    score = v_norm + z_weight * z_norm
    idx = int(np.argmin(score))
    return start + idx


def segment_single_cycle(
    T: int,
    vel: np.ndarray,
    z: np.ndarray,
    t_close: int,
    t_open: int,
    window: int = 2,
) -> List[Segment]:
    """
    Build segments for a single pick-and-place cycle.
    """
    segments: List[Segment] = []

    # clamp indices
    t_close = int(np.clip(t_close, 0, T - 1))
    t_open = int(np.clip(t_open, 0, T - 1))
    if t_open <= t_close:
        # degenerate, treat as single grasp-like event
        segments.append(Segment(0, "move_to_object", 0, max(0, t_close - window)))
        segments.append(Segment(1, "grasp_object", max(0, t_close - window), min(T - 1, t_close + window)))
        return segments

    # initial coarse ranges
    s0_start, s0_end = 0, max(0, t_close - window)
    s1_start, s1_end = max(0, t_close - window), min(T - 1, t_close + window)
    s2_start, s2_end = min(T - 1, t_close + window), max(s1_end + 1, t_open - window)
    s3_start, s3_end = max(s2_end + 1, t_open - window), min(T - 1, t_open + window)
    s4_start, s4_end = min(T - 1, t_open + window), T - 1

    # refine centers for grasp/place to be low-velocity, low-height
    grasp_center = refine_center_by_velocity_and_height(vel, z, s1_start, s1_end)
    place_center = refine_center_by_velocity_and_height(vel, z, s3_start, s3_end)

    # shrink around centers to avoid ultra-wide segments
    half_span = max(1, window)
    s1_start, s1_end = max(0, grasp_center - half_span), min(T - 1, grasp_center + half_span)
    s3_start, s3_end = max(0, place_center - half_span), min(T - 1, place_center + half_span)

    # recompute dependent ranges with simple monotonic constraints
    s0_start, s0_end = 0, max(0, s1_start - 1)
    s2_start = min(T - 1, s1_end + 1)
    s2_end = max(s2_start, s3_start - 1)
    s4_start, s4_end = min(T - 1, s3_end + 1), T - 1

    raw_segments = [
        ("move_to_object", s0_start, s0_end),
        ("grasp_object", s1_start, s1_end),
        ("move_to_goal", s2_start, s2_end),
        ("place_object", s3_start, s3_end),
        ("retract", s4_start, s4_end),
    ]

    # filter degenerate/very short segments by merging into neighbors
    cleaned: List[Tuple[str, int, int]] = []
    min_len = 2
    for seg_type, start, end in raw_segments:
        if end < start:
            continue
        if end - start + 1 < min_len and cleaned:
            # merge into previous segment
            prev_type, prev_start, prev_end = cleaned[-1]
            cleaned[-1] = (prev_type, prev_start, end)
        else:
            cleaned.append((seg_type, start, end))

    segments = [
        Segment(segment_id=i, segment_type=seg_type, start_t=start, end_t=end)
        for i, (seg_type, start, end) in enumerate(cleaned)
    ]
    return segments


def build_per_step_labels(T: int, cycles: List[Cycle]) -> Dict[str, List[int]]:
    subtask_ids = [0] * T
    cycle_ids = [0] * T

    for cycle in cycles:
        for seg in cycle.segments:
            for t in range(seg.start_t, seg.end_t + 1):
                if 0 <= t < T:
                    subtask_ids[t] = SEGMENT_TYPES.index(seg.segment_type)
                    cycle_ids[t] = cycle.cycle_id

    return {
        "subtask_id": subtask_ids,
        "cycle_id": cycle_ids,
    }


def segment_episode(
    episode_index: int,
    root: Path,
    chunk_size: int,
    th_open: float = 0.3,
    th_close: float = 0.7,
    window: int = 2,
) -> Dict[str, Any]:
    data = load_episode_parquet(root, episode_index, chunk_size)
    state, action = data["state"], data["action"]
    T = state.shape[0]

    feats = derive_features(state, action)
    vel, z, g_raw = feats["vel"], feats["z"], feats["gripper_raw"]

    g_bin = binarize_gripper(g_raw, th_open=th_open, th_close=th_close)
    t_close_list, t_open_list = detect_gripper_events(g_bin)

    cycles: List[Cycle] = []
    segmentation_quality = "low"

    if t_close_list and t_open_list:
        # simple pairing: for each close, find the first open after it
        used_opens = set()
        cycle_id = 0
        for t_close in t_close_list:
            t_open_candidates = [t for t in t_open_list if t > t_close and t not in used_opens]
            if not t_open_candidates:
                continue
            t_open = t_open_candidates[0]
            used_opens.add(t_open)

            segments = segment_single_cycle(T, vel, z, t_close, t_open, window=window)
            cycles.append(Cycle(cycle_id=cycle_id, segments=segments))
            cycle_id += 1

        if cycles:
            segmentation_quality = "high"
    else:
        # no clear pick-and-place events detected
        cycles = []
        segmentation_quality = "low"

    per_step_labels = build_per_step_labels(T, cycles)

    result: Dict[str, Any] = {
        "dataset_root": str(root),
        "episode_index": episode_index,
        "num_steps": int(T),
        "segmentation_quality": segmentation_quality,
        "cycles": [
            {
                "cycle_id": cycle.cycle_id,
                "segments": [asdict(seg) for seg in cycle.segments],
            }
            for cycle in cycles
        ],
        "per_step_labels": per_step_labels,
    }
    return result


def load_info(root: Path) -> Dict[str, Any]:
    info_path = root / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_episode_indices(root: Path, split: str = "train") -> List[int]:
    info = load_info(root)
    split_spec = info.get("splits", {}).get(split)
    if split_spec is None:
        raise ValueError(f"Split '{split}' not found in info.json")
    # split_spec format: "start:end"
    start_str, end_str = split_spec.split(":")
    start, end = int(start_str), int(end_str)
    return list(range(start, end))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-based trajectory segmentation for BRIDGE-LeRobot.")
    parser.add_argument(
        "--bridge_root",
        type=str,
        default=BRIDGE_ROOT_DEFAULT,
        help="Root directory of bridge_orig_lerobot dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split name defined in info.json (e.g., train).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="segmentation.jsonl",
        help="Output JSONL file path (will be created/overwritten).",
    )
    parser.add_argument(
        "--th_open",
        type=float,
        default=0.3,
        help="Threshold for gripper 'open' when binarizing.",
    )
    parser.add_argument(
        "--th_close",
        type=float,
        default=0.7,
        help="Threshold for gripper 'close' when binarizing.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=2,
        help="Half-window size around gripper events for grasp/place segments.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=-1,
        help="If >0, limit the number of episodes processed (for debugging).",
    )

    args = parser.parse_args()
    root = Path(args.bridge_root)
    info = load_info(root)
    chunk_size = int(info["chunks_size"])

    episode_indices = iter_episode_indices(root, split=args.split)
    if args.max_episodes > 0:
        episode_indices = episode_indices[: args.max_episodes]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for idx, ep in enumerate(episode_indices):
            result = segment_episode(
                episode_index=ep,
                root=root,
                chunk_size=chunk_size,
                th_open=args.th_open,
                th_close=args.th_close,
                window=args.window,
            )
            f.write(json.dumps(result) + "\n")

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} episodes...")

    print(f"Segmentation finished. Saved to {output_path}")


if __name__ == "__main__":
    main()

