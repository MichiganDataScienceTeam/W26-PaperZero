"""
Training
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
import faulthandler
import gc
import multiprocessing as mp
import os
from pathlib import Path
from time import perf_counter
import traceback
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.origami_sampler import OrigamiSampler
from mcts.encoder import ThinkArchitecture
from mcts.node import Node, RASTER_RESOLUTION
from paper import Paper


# Training loop constants
REPLAY_BUFFER_SIZE = 2048
NUM_CYCLES = 1000
TRAIN_EPOCHS_PER_CYCLE = 1
BATCH_SIZE = 32
START_MAX_FOLD_LEVEL = 2
HARD_MAX_FOLD_LEVEL = 4
LOWER_LEVEL_MIX_FRACTION = 0.30
LEVEL_CHANGE_PATIENCE_CYCLES = 3
LEVEL_CHANGE_MIN_DELTA = 1e-6
ROLLOUTS_PER_ROOT = 16
MAX_TREE_DEPTH = 10
TREE_IOU_TERMINATION_THRESHOLD = 0.9
MIN_NODE_VISITS_FOR_RECORD = 3
NEW_RECORDS_PER_CYCLE = 256
HELDOUT_TARGET_COUNT = 10
EVAL_ROLLOUTS_PER_TARGET = 8
PERF_SAMPLE_COUNT = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4
ENABLE_HELDOUT_EVAL = True
SAVE_CHECKPOINTS = True
CHECKPOINT_DIR = "mcts/checkpoints"

# Multiprocessing data pipeline constants
ENABLE_MULTIPROCESS_DATA_GEN = True
NUM_DATA_GEN_WORKERS = min(8, max(1, (os.cpu_count() or 2) // 2))
DATA_GEN_MAX_TASKS_PER_WORKER = 32
DATA_GEN_DIAG_DIR = "mcts/checkpoints/datagen_diagnostics"
DATA_GEN_MAX_POOL_RESTARTS = 6
DATA_GEN_MAX_ROOT_ATTEMPTS_MULTIPLIER = 20
DATA_GEN_MAX_EMPTY_ROOT_STREAK = 50

# Performance constants
ENABLE_CUDA_AMP = True
DATA_LOADER_NUM_WORKERS = 0

_WORKER_MODEL: ThinkArchitecture | None = None
_MODEL: ThinkArchitecture | None = None


def get_model() -> ThinkArchitecture:
    if _MODEL is None:
        raise RuntimeError("MCTS model has not been initialized. Run main() first.")
    return _MODEL


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sample_training_level(current_max_level: int) -> int:
    if current_max_level <= 1:
        return 1
    if np.random.random() >= LOWER_LEVEL_MIX_FRACTION:
        return current_max_level
    return int(np.random.randint(1, current_max_level))


def _format_level_counts(level_counts: dict[int, int]) -> str:
    if len(level_counts) == 0:
        return "{}"
    ordered = sorted(level_counts.items(), key=lambda item: item[0])
    return "{" + ", ".join(f"L{level}:{count}" for level, count in ordered) + "}"


class FoldDataset(Dataset):
    H = W = RASTER_RESOLUTION

    def __init__(self, records: list[dict[str, Any]]):
        self.samples = []
        for rec in records:
            state = np.array(rec["state"], dtype=np.float32)
            policy_target = np.array(rec["policy_target"], dtype=np.float32)
            value = float(rec["value"])
            if state.shape != (2, self.H, self.W):
                raise ValueError(f"bad state shape: {state.shape}")
            if policy_target.shape != (2, self.H, self.W):
                raise ValueError(f"bad policy_target shape: {policy_target.shape}")

            state_tensor = torch.tensor(state, dtype=torch.float32)
            policy_tensor = torch.tensor(policy_target, dtype=torch.float32)
            value_tensor = torch.tensor([value], dtype=torch.float32)
            self.samples.append((state_tensor, policy_tensor, value_tensor))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _make_root_node(sampler: OrigamiSampler, model: ThinkArchitecture, level: int) -> Node:
    sample = sampler.sample(level=level)
    final_paper = sample["final_paper"]
    target_mask = np.array(
        final_paper.rasterize(RASTER_RESOLUTION, RASTER_RESOLUTION, 0.0),
        dtype=np.float32,
    ).reshape(RASTER_RESOLUTION, RASTER_RESOLUTION)
    return Node(paper=Paper(), model=model, target_mask=target_mask)


def _make_root_from_target(target_mask: np.ndarray, model: ThinkArchitecture) -> Node:
    return Node(paper=Paper(), model=model, target_mask=np.array(target_mask, dtype=np.float32))


def _calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    mask1b = mask1.astype(bool)
    mask2b = mask2.astype(bool)
    intersection = np.logical_and(mask1b, mask2b).sum()
    union = np.logical_or(mask1b, mask2b).sum()
    return float(intersection / union) if union > 0 else 0.0


def _is_terminal_leaf(node: Node, cache: dict[int, bool]) -> bool:
    key = id(node)
    if key in cache:
        return cache[key]
    node_mask = np.array(
        node.paper.rasterize(RASTER_RESOLUTION, RASTER_RESOLUTION, 0.0),
        dtype=np.float32,
    ).reshape(RASTER_RESOLUTION, RASTER_RESOLUTION)
    terminal = _calculate_iou(node_mask, np.array(node.target_mask, dtype=np.float32)) >= TREE_IOU_TERMINATION_THRESHOLD
    cache[key] = terminal
    return terminal


def _batched_expand_nodes(nodes: list[Node], model: ThinkArchitecture, device: torch.device) -> tuple[int, int]:
    if len(nodes) == 0:
        return 0, 0

    model.eval()
    b = len(nodes)
    h = RASTER_RESOLUTION
    w = RASTER_RESOLUTION
    states = np.empty((b, 2, h, w), dtype=np.float32)

    for i, node in enumerate(nodes):
        states[i, 0] = np.array(node.paper.rasterize(h, w, 0.0), dtype=np.float32).reshape(h, w)
        states[i, 1] = node.target_mask

    with torch.inference_mode():
        input_batch = torch.from_numpy(states).to(device, non_blocking=(device.type == "cuda"))
        policy_batch, value_batch = model(input_batch)

    expanded_nodes = 0
    for i, node in enumerate(nodes):
        node._expand_from_network_outputs(policy_batch[i], float(value_batch[i].item()))
        if node.was_expanded:
            expanded_nodes += 1

    return len(nodes), expanded_nodes


def _run_rollouts(root: Node, num_rollouts: int, model: ThinkArchitecture, device: torch.device) -> dict[str, int | float]:
    expand_calls = 0
    expanded_nodes = 0
    select_calls = 0
    terminal_cache: dict[int, bool] = {}
    started = perf_counter()
    nodes: list[Node | None] = [root for _ in range(num_rollouts)]
    depths = [0 for _ in range(num_rollouts)]

    while True:
        if all(node is None for node in nodes):
            break

        frontier: dict[int, Node] = {}
        for i, node in enumerate(nodes):
            if node is None:
                continue
            if depths[i] >= MAX_TREE_DEPTH:
                nodes[i] = None
                continue
            if len(node.children) == 0:
                if _is_terminal_leaf(node, terminal_cache):
                    nodes[i] = None
                    continue
                frontier[id(node)] = node

        if frontier:
            calls, expanded = _batched_expand_nodes(list(frontier.values()), model, device)
            expand_calls += calls
            expanded_nodes += expanded

        for i, node in enumerate(nodes):
            if node is None:
                continue
            select_calls += 1
            nxt = node.select()
            if nxt is None:
                nodes[i] = None
                continue
            nodes[i] = nxt
            depths[i] += 1

        if max(depths, default=0) >= MAX_TREE_DEPTH and all(
            (node is None or depth >= MAX_TREE_DEPTH) for node, depth in zip(nodes, depths)
        ):
            break

    return {
        "seconds": perf_counter() - started,
        "expand_calls": expand_calls,
        "expanded_nodes": expanded_nodes,
        "select_calls": select_calls,
    }


def _collect_training_records(root: Node) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    stack = [root]
    while stack:
        node = stack.pop()
        stack.extend(node.children)

        if not node.was_expanded:
            continue
        if node.N < MIN_NODE_VISITS_FOR_RECORD:
            continue

        record = node.export_training_record()
        records.append(record)
    return records


def _worker_init(model_state_dict: dict[str, torch.Tensor]) -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    os.makedirs(DATA_GEN_DIAG_DIR, exist_ok=True)
    crash_log_path = os.path.join(DATA_GEN_DIAG_DIR, f"worker_{os.getpid()}.fault.log")
    crash_file = open(crash_log_path, "a", buffering=1)
    faulthandler.enable(file=crash_file, all_threads=True)
    globals()["_WORKER_CRASH_FILE"] = crash_file

    model = ThinkArchitecture(2, 128, 3)
    model.load_state_dict(model_state_dict, strict=True)
    model.eval()

    global _WORKER_MODEL
    _WORKER_MODEL = model


def _worker_generate_one_root(current_max_level: int) -> dict[str, Any]:
    try:
        if _WORKER_MODEL is None:
            raise RuntimeError("worker model is not initialized")
        sampler = OrigamiSampler()
        model = _WORKER_MODEL
        sampled_level = _sample_training_level(current_max_level)
        root = _make_root_node(sampler, model, sampled_level)
        rollout_stats = _run_rollouts(root, ROLLOUTS_PER_ROOT, model, torch.device("cpu"))
        records = _collect_training_records(root)
        return {
            "records": records,
            "rollout_seconds": float(rollout_stats["seconds"]),
            "expand_calls": int(rollout_stats["expand_calls"]),
            "sampled_level": sampled_level,
        }
    except Exception:
        os.makedirs(DATA_GEN_DIAG_DIR, exist_ok=True)
        err_path = os.path.join(DATA_GEN_DIAG_DIR, f"worker_{os.getpid()}.python_error.log")
        with open(err_path, "a") as fh:
            fh.write(traceback.format_exc())
            fh.write("\n")
        raise
    finally:
        gc.collect()


def _new_data_gen_executor(model: ThinkArchitecture, max_workers: int) -> ProcessPoolExecutor:
    model_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ctx = mp.get_context("spawn")
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(model_state_dict,),
        max_tasks_per_child=DATA_GEN_MAX_TASKS_PER_WORKER,
    )


def _generate_records_for_target_sequential(
    target_records: int,
    model: ThinkArchitecture,
    device: torch.device,
    current_max_level: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_records: list[dict[str, Any]] = []
    total_rollout_seconds = 0.0
    total_expand_calls = 0
    roots_attempted = 0
    empty_root_streak = 0
    level_counts: dict[int, int] = {}
    max_root_attempts = max(1, target_records * DATA_GEN_MAX_ROOT_ATTEMPTS_MULTIPLIER)
    sampler = OrigamiSampler()

    while len(all_records) < target_records:
        if roots_attempted >= max_root_attempts:
            raise RuntimeError(
                f"Sequential datagen hit root-attempt limit ({roots_attempted}) with only "
                f"{len(all_records)}/{target_records} records."
            )

        roots_attempted += 1
        sampled_level = _sample_training_level(current_max_level)
        level_counts[sampled_level] = level_counts.get(sampled_level, 0) + 1
        root = _make_root_node(sampler, model, sampled_level)
        rollout_stats = _run_rollouts(root, ROLLOUTS_PER_ROOT, model, device)
        records = _collect_training_records(root)

        total_rollout_seconds += float(rollout_stats["seconds"])
        total_expand_calls += int(rollout_stats["expand_calls"])

        if len(records) == 0:
            empty_root_streak += 1
            if empty_root_streak >= DATA_GEN_MAX_EMPTY_ROOT_STREAK:
                raise RuntimeError(
                    "Sequential datagen produced too many empty roots in a row "
                    f"({empty_root_streak})."
                )
            continue

        empty_root_streak = 0
        needed = target_records - len(all_records)
        all_records.extend(records[:needed])

    stats: dict[str, Any] = {
        "records": len(all_records),
        "roots_attempted": roots_attempted,
        "roots_failed": 0,
        "pool_restarts": 0,
        "rollout_seconds": total_rollout_seconds,
        "expand_calls": total_expand_calls,
        "level_counts": level_counts,
    }
    return all_records, stats


def _generate_records_for_target_multiprocess(
    target_records: int,
    model: ThinkArchitecture,
    current_max_level: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    workers = min(NUM_DATA_GEN_WORKERS, max(1, target_records))
    all_records: list[dict[str, Any]] = []
    max_root_attempts = max(1, target_records * DATA_GEN_MAX_ROOT_ATTEMPTS_MULTIPLIER)
    root_attempts = 0
    root_failures = 0
    pool_restarts = 0
    empty_root_streak = 0
    level_counts: dict[int, int] = {}
    total_rollout_seconds = 0.0
    total_expand_calls = 0

    pool = _new_data_gen_executor(model, workers)
    pending: set[Future] = set()

    def _submit_one() -> None:
        nonlocal root_attempts
        if root_attempts >= max_root_attempts:
            return
        pending.add(pool.submit(_worker_generate_one_root, current_max_level))
        root_attempts += 1

    try:
        while len(all_records) < target_records:
            while len(pending) < workers and root_attempts < max_root_attempts and len(all_records) < target_records:
                _submit_one()

            if len(pending) == 0:
                break

            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            pool_was_restarted = False

            for fut in done:
                pending.remove(fut)
                try:
                    payload = fut.result()
                except BrokenProcessPool as exc:
                    pool_restarts += 1
                    print(
                        f"[datagen] pool broke (restart={pool_restarts}/{DATA_GEN_MAX_POOL_RESTARTS}): {exc}"
                    )
                    if pool_restarts > DATA_GEN_MAX_POOL_RESTARTS:
                        raise RuntimeError(
                            "Multiprocessing datagen exceeded pool restart budget."
                        ) from exc

                    for other in pending:
                        other.cancel()
                    pending.clear()
                    pool.shutdown(wait=True, cancel_futures=True)
                    pool = _new_data_gen_executor(model, workers)
                    pool_was_restarted = True
                    break
                except Exception as exc:
                    root_failures += 1
                    print(f"[datagen] worker root failed (count={root_failures}): {type(exc).__name__}: {exc}")
                    continue

                root_records = payload.get("records", [])
                total_rollout_seconds += float(payload.get("rollout_seconds", 0.0))
                total_expand_calls += int(payload.get("expand_calls", 0))
                sampled_level = int(payload.get("sampled_level", current_max_level))
                level_counts[sampled_level] = level_counts.get(sampled_level, 0) + 1

                if len(root_records) == 0:
                    empty_root_streak += 1
                    if empty_root_streak >= DATA_GEN_MAX_EMPTY_ROOT_STREAK:
                        raise RuntimeError(
                            "Multiprocessing datagen produced too many empty roots in a row "
                            f"({empty_root_streak})."
                        )
                    continue

                empty_root_streak = 0
                needed = target_records - len(all_records)
                all_records.extend(root_records[:needed])

            if pool_was_restarted:
                continue

        if len(all_records) < target_records:
            raise RuntimeError(
                "Multiprocessing datagen hit root-attempt limit with insufficient records: "
                f"{len(all_records)}/{target_records}"
            )
    finally:
        for fut in pending:
            fut.cancel()
        pool.shutdown(wait=True, cancel_futures=True)

    stats: dict[str, Any] = {
        "records": len(all_records),
        "roots_attempted": root_attempts,
        "roots_failed": root_failures,
        "pool_restarts": pool_restarts,
        "rollout_seconds": total_rollout_seconds,
        "expand_calls": total_expand_calls,
        "level_counts": level_counts,
    }
    return all_records, stats


def _generate_records_for_target(
    target_records: int,
    model: ThinkArchitecture,
    device: torch.device,
    current_max_level: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target_records <= 0:
        return [], {
            "records": 0,
            "roots_attempted": 0,
            "roots_failed": 0,
            "pool_restarts": 0,
            "rollout_seconds": 0.0,
            "expand_calls": 0,
            "level_counts": {},
        }

    use_mp = ENABLE_MULTIPROCESS_DATA_GEN and NUM_DATA_GEN_WORKERS > 1
    if use_mp:
        return _generate_records_for_target_multiprocess(target_records, model, current_max_level)
    return _generate_records_for_target_sequential(target_records, model, device, current_max_level)


def _train_epochs(
    data_loader: DataLoader,
    model: ThinkArchitecture,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
) -> tuple[float, float]:
    policy_criterion = torch.nn.MSELoss()
    value_criterion = torch.nn.MSELoss()
    use_amp = ENABLE_CUDA_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    non_blocking = device.type == "cuda"

    started = perf_counter()
    last_avg_loss = float("inf")
    for epoch_idx in range(epochs):
        model.train()
        total_loss = 0.0
        for state, policy_target, value_target in data_loader:
            state = state.to(device, non_blocking=non_blocking)
            policy_target = policy_target.to(device, non_blocking=non_blocking)
            value_target = value_target.to(device, non_blocking=non_blocking)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                policy_pred, value_pred = model(state)
                policy_loss = policy_criterion(policy_pred, policy_target)
                value_loss = value_criterion(value_pred, value_target)
                loss = policy_loss + value_loss

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(len(data_loader), 1)
        last_avg_loss = avg_loss
        assert np.isfinite(avg_loss), f"non-finite training loss: {avg_loss}"
        print(f"    epoch {epoch_idx + 1}/{epochs} avg loss: {avg_loss:.6f}")
    return perf_counter() - started, last_avg_loss


def _fill_replay_buffer(
    buffer: deque[dict[str, Any]],
    model: ThinkArchitecture,
    device: torch.device,
    current_max_level: int,
) -> None:
    while len(buffer) < REPLAY_BUFFER_SIZE:
        needed = REPLAY_BUFFER_SIZE - len(buffer)
        records, stats = _generate_records_for_target(needed, model, device, current_max_level)

        for record in records:
            if len(buffer) >= REPLAY_BUFFER_SIZE:
                break
            buffer.append(record)

        print(
            f"[fill] added={len(records)} buffer={len(buffer)}/{REPLAY_BUFFER_SIZE} "
            f"roots={int(stats['roots_attempted'])} failed={int(stats['roots_failed'])} "
            f"restarts={int(stats['pool_restarts'])} rollout_s={float(stats['rollout_seconds']):.2f} "
            f"expand_calls={int(stats['expand_calls'])} levels={_format_level_counts(stats['level_counts'])}"
        )

    assert len(buffer) == REPLAY_BUFFER_SIZE, f"buffer fill failed: {len(buffer)}"


def _profile_runtime_basics(
    sampler: OrigamiSampler,
    model: ThinkArchitecture,
    device: torch.device,
    current_max_level: int,
) -> None:
    sample_times: list[float] = []
    for _ in range(PERF_SAMPLE_COUNT):
        t0 = perf_counter()
        sampled_level = _sample_training_level(current_max_level)
        _ = sampler.sample(level=sampled_level)
        sample_times.append(perf_counter() - t0)

    model.eval()
    dummy = torch.zeros((1, 2, RASTER_RESOLUTION, RASTER_RESOLUTION), dtype=torch.float32, device=device)
    forward_times: list[float] = []
    with torch.no_grad():
        for _ in range(PERF_SAMPLE_COUNT):
            t0 = perf_counter()
            _ = model(dummy)
            forward_times.append(perf_counter() - t0)

    print(
        "[perf] sampler.sample mean={:.4f}s p95={:.4f}s | model.forward mean={:.4f}s p95={:.4f}s".format(
            float(np.mean(sample_times)),
            float(np.percentile(sample_times, 95)),
            float(np.mean(forward_times)),
            float(np.percentile(forward_times, 95)),
        )
    )


def _best_iou_in_tree(root: Node) -> float:
    target = np.array(root.target_mask, dtype=np.float32)
    best = 0.0
    stack = [root]
    while stack:
        node = stack.pop()
        stack.extend(node.children)
        node_mask = np.array(
            node.paper.rasterize(RASTER_RESOLUTION, RASTER_RESOLUTION, 0.0),
            dtype=np.float32,
        ).reshape(RASTER_RESOLUTION, RASTER_RESOLUTION)
        iou = _calculate_iou(node_mask, target)
        if iou > best:
            best = iou
    return best


def _evaluate_holdout_iou(
    holdout_targets: list[np.ndarray],
    model: ThinkArchitecture,
    device: torch.device,
) -> dict[str, float]:
    ious: list[float] = []
    eval_rollout_seconds = 0.0
    for target in holdout_targets:
        root = _make_root_from_target(target, model)
        rollout_stats = _run_rollouts(root, EVAL_ROLLOUTS_PER_TARGET, model, device)
        eval_rollout_seconds += float(rollout_stats["seconds"])
        ious.append(_best_iou_in_tree(root))

    arr = np.array(ious, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std()),
        "eval_rollout_seconds": eval_rollout_seconds,
    }


def _save_checkpoint(model: ThinkArchitecture, optimizer: torch.optim.Optimizer, cycle: int | None) -> Path:
    out_dir = Path(CHECKPOINT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = "model_final.pt" if cycle is None else f"model_cycle_{cycle:03d}.pt"
    path = out_dir / filename
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cycle": cycle,
        },
        path,
    )
    return path


def main() -> None:
    Path(DATA_GEN_DIAG_DIR).mkdir(parents=True, exist_ok=True)
    device = _get_device()
    print(f"[device] using {device}")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[perf] CUDA optimizations enabled: cudnn.benchmark + TF32 + AMP")

    model = ThinkArchitecture(2, 128, 3)
    model.to(device)
    global _MODEL
    _MODEL = model

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sampler = OrigamiSampler()
    current_max_level = START_MAX_FOLD_LEVEL
    _profile_runtime_basics(sampler, model, device, current_max_level)
    print(
        f"[datagen] multiprocessing={ENABLE_MULTIPROCESS_DATA_GEN} workers={NUM_DATA_GEN_WORKERS} "
        f"level_mix=({int(LOWER_LEVEL_MIX_FRACTION * 100)}% lower, {100 - int(LOWER_LEVEL_MIX_FRACTION * 100)}% max)"
    )

    holdout_targets: list[np.ndarray] = []
    if ENABLE_HELDOUT_EVAL:
        for _ in range(HELDOUT_TARGET_COUNT):
            sample = sampler.sample(level=current_max_level)
            final_paper = sample["final_paper"]
            target_mask = np.array(
                final_paper.rasterize(RASTER_RESOLUTION, RASTER_RESOLUTION, 0.0),
                dtype=np.float32,
            ).reshape(RASTER_RESOLUTION, RASTER_RESOLUTION)
            holdout_targets.append(target_mask)

    buffer: deque[dict[str, Any]] = deque(maxlen=REPLAY_BUFFER_SIZE)
    best_loss_for_setup = float("inf")
    stale_loss_cycles = 0

    _fill_replay_buffer(buffer, model, device, current_max_level)
    print(f"[fill] completed buffer={len(buffer)}")

    for cycle in range(1, NUM_CYCLES + 1):
        cycle_start = perf_counter()
        print(f"[cycle {cycle}] step 1: train (max_level={current_max_level})")
        data_start = perf_counter()
        dataset = FoldDataset(list(buffer))
        data_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=DATA_LOADER_NUM_WORKERS,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(DATA_LOADER_NUM_WORKERS > 0),
        )
        data_seconds = perf_counter() - data_start
        train_seconds, avg_loss = _train_epochs(data_loader, model, optimizer, device, TRAIN_EPOCHS_PER_CYCLE)

        if avg_loss + LEVEL_CHANGE_MIN_DELTA < best_loss_for_setup:
            best_loss_for_setup = avg_loss
            stale_loss_cycles = 0
        else:
            stale_loss_cycles += 1

        if stale_loss_cycles >= LEVEL_CHANGE_PATIENCE_CYCLES:
            if current_max_level < HARD_MAX_FOLD_LEVEL:
                previous_level = current_max_level
                current_max_level += 1
                best_loss_for_setup = float("inf")
                stale_loss_cycles = 0
                print(
                    f"[cycle {cycle}] curriculum level update: max_level {previous_level} -> {current_max_level} "
                    f"(no new low loss for {LEVEL_CHANGE_PATIENCE_CYCLES} cycles)"
                )
            else:
                stale_loss_cycles = 0
                print(
                    f"[cycle {cycle}] curriculum level held at cap max_level={current_max_level} "
                    f"(no new low loss for {LEVEL_CHANGE_PATIENCE_CYCLES} cycles)"
                )

        print(f"[cycle {cycle}] step 2: collect {NEW_RECORDS_PER_CYCLE} new records")
        records, stats = _generate_records_for_target(NEW_RECORDS_PER_CYCLE, model, device, current_max_level)
        for record in records:
            buffer.append(record)

        print(
            f"[cycle {cycle}] added_records={len(records)} buffer_size={len(buffer)} "
            f"roots={int(stats['roots_attempted'])} failed={int(stats['roots_failed'])} "
            f"restarts={int(stats['pool_restarts'])} levels={_format_level_counts(stats['level_counts'])}"
        )
        assert len(buffer) == REPLAY_BUFFER_SIZE, "buffer size changed after FIFO update"

        eval_rollout_seconds = 0.0
        if ENABLE_HELDOUT_EVAL:
            print(f"[cycle {cycle}] step 3: heldout testing ({HELDOUT_TARGET_COUNT} targets)")
            eval_stats = _evaluate_holdout_iou(holdout_targets, model, device)
            eval_rollout_seconds = float(eval_stats["eval_rollout_seconds"])
            print(
                f"[cycle {cycle}] iou mean={eval_stats['mean']:.4f} median={eval_stats['median']:.4f} "
                f"min={eval_stats['min']:.4f} max={eval_stats['max']:.4f} std={eval_stats['std']:.4f}"
            )
        else:
            print(f"[cycle {cycle}] step 3: heldout testing skipped (ENABLE_HELDOUT_EVAL=False)")

        if SAVE_CHECKPOINTS:
            checkpoint_path = _save_checkpoint(model, optimizer, cycle)
            print(f"[cycle {cycle}] saved checkpoint: {checkpoint_path}")

        cycle_seconds = perf_counter() - cycle_start
        print(
            f"[perf cycle {cycle}] avg_loss={avg_loss:.6f} data_s={data_seconds:.2f} train_s={train_seconds:.2f} "
            f"rollout_s={float(stats['rollout_seconds']):.2f} eval_rollout_s={eval_rollout_seconds:.2f} "
            f"expand_calls={int(stats['expand_calls'])} total_s={cycle_seconds:.2f}"
        )

    if SAVE_CHECKPOINTS:
        final_path = _save_checkpoint(model, optimizer, None)
        print(f"[done] saved final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
