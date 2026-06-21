from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from paper import Paper, Segment, Vec2

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from mcts.encoder import ThinkArchitecture


RESOLUTION = 128
CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
FIXED_CHECKPOINT_NAME = "model_cycle_068.pt"
FIXED_CHECKPOINT_ROOT_PATH = Path(__file__).resolve().parents[2] / FIXED_CHECKPOINT_NAME
app = Flask(__name__)


@dataclass
class _BlankNode:
    N: int = 0
    W: float = 0.0
    Q: float = 0.0
    P: float = 1.0
    paper: Any = field(default_factory=Paper)
    parent: Any | None = None
    children: list[Any] = field(default_factory=list)


@dataclass
class _RuntimeNode:
    paper: Any
    parent: Any | None = None
    N: int = 0
    W: float = 0.0
    Q: float = 0.0
    P: float = 0.0
    children: list[Any] = field(default_factory=list)


@dataclass
class _DemoNode:
    N: int = 0
    W: float = 0.0
    Q: float = 0.0
    P: float = 0.0
    paper: Any | None = None
    parent: Any | None = None
    children: list[Any] = field(default_factory=list)


@dataclass
class _VisualizerState:
    tree_root: Any | None = None
    default_root: Any | None = None
    network: ThinkArchitecture | None = None
    device: torch.device | None = None
    loaded_checkpoint: str | None = None


_STATE = _VisualizerState()


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _empty_mask() -> np.ndarray:
    return np.zeros((RESOLUTION, RESOLUTION), dtype=np.float32)


def _make_default_root() -> Any:
    try:
        from mcts.node import Node
        net = _ensure_network()
        return Node(Paper(), model=net, target_mask=np.ones((RESOLUTION, RESOLUTION), dtype=np.float32))
    except Exception:
        return _BlankNode()


def set_tree_root(root: Any | None) -> None:
    _STATE.tree_root = root


def _current_root() -> Any:
    if _STATE.tree_root is not None:
        return _STATE.tree_root
    if _STATE.default_root is None:
        _STATE.default_root = _make_default_root()
    return _STATE.default_root


def _is_visited(node: Any) -> bool:
    return int(getattr(node, "N", 0)) > 0


def _ensure_network() -> ThinkArchitecture:
    if _STATE.network is None:
        _STATE.device = _get_device()
        net = ThinkArchitecture(2, 128, 3)
        net.to(_STATE.device)
        net.eval()
        _STATE.network = net
        _load_checkpoint(_resolve_fixed_checkpoint())
    return _STATE.network


def _paper_mask(paper: Any | None) -> np.ndarray:
    if paper is None:
        return _empty_mask()
    raster = np.array(paper.rasterize(RESOLUTION, RESOLUTION, 0.0), dtype=np.float32)
    return raster.reshape(RESOLUTION, RESOLUTION)


def _target_mask(node: Any) -> np.ndarray:
    raw = getattr(node, "target_mask", None)
    if raw is None:
        return np.ones((RESOLUTION, RESOLUTION), dtype=np.float32)

    arr = np.array(raw, dtype=np.float32)
    if arr.shape != (RESOLUTION, RESOLUTION):
        try:
            arr = arr.reshape(RESOLUTION, RESOLUTION)
        except Exception:
            arr = np.ones((RESOLUTION, RESOLUTION), dtype=np.float32)
    return arr


def _paper_bounds(paper: Any | None) -> list[float]:
    if paper is None:
        return [0.0, 1.0, 0.0, 1.0]

    try:
        bounds = paper.compute_bounds()
        if len(bounds) == 4:
            min_x, max_x, min_y, max_y = [float(v) for v in bounds]
            if max_x > min_x and max_y > min_y:
                return [min_x, max_x, min_y, max_y]
    except Exception:
        pass

    return [0.0, 1.0, 0.0, 1.0]


def _nn_outputs(mask: np.ndarray, target_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    net = _ensure_network()
    device = next(net.parameters()).device

    with torch.no_grad():
        stacked = np.stack([mask, target_mask], axis=0)
        image = torch.tensor(stacked, dtype=torch.float32, device=device).unsqueeze(0)
        policy, value = net(image)
        policy = policy.squeeze(0)
        start_logits = policy[0]
        end_logits = policy[1]
        start_probs = torch.softmax(start_logits.flatten(), dim=0).view_as(start_logits)
        end_probs = torch.softmax(end_logits.flatten(), dim=0).view_as(end_logits)

    return (
        start_probs.detach().cpu().numpy().astype(np.float32),
        end_probs.detach().cpu().numpy().astype(np.float32),
        float(value.item()),
    )


def _to_tree_view(root: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    node_lookup: dict[str, Any] = {}

    def build(node: Any, node_id: str, force_include: bool = False) -> dict[str, Any] | None:
        visited = _is_visited(node)
        if not visited and not force_include:
            return None

        children = getattr(node, "children", []) or []
        tree_children: list[dict[str, Any]] = []
        visited_children_count = 0

        # Keep stable IDs by preserving original child index.
        for idx, child in enumerate(children):
            if not _is_visited(child):
                continue
            child_id = f"{node_id}.{idx}"
            child_view = build(child, child_id)
            if child_view is not None:
                tree_children.append(child_view)
                visited_children_count += 1

        node_lookup[node_id] = node
        paper = getattr(node, "paper", None)
        preview = _paper_mask(paper)

        return {
            "id": node_id,
            "visits": int(getattr(node, "N", 0)),
            "value_sum": float(getattr(node, "W", 0.0)),
            "q": float(getattr(node, "Q", 0.0)),
            "prior": float(getattr(node, "P", 0.0)),
            "unvisited_children": len(children) - visited_children_count,
            "bounds": _paper_bounds(paper),
            "preview_mask": preview.tolist(),
            "children": tree_children,
        }

    tree = build(root, "root", force_include=True)
    assert tree is not None
    return tree, node_lookup


def _resolve_node(node_id: str) -> Any | None:
    root = _current_root()
    _, lookup = _to_tree_view(root)
    return lookup.get(node_id)


def _node_payload(node: Any, node_id: str) -> dict[str, Any]:
    paper = getattr(node, "paper", None)
    mask = _paper_mask(paper)
    target_mask = _target_mask(node)
    start_map, end_map, value_estimate = _nn_outputs(mask, target_mask)

    return {
        "id": node_id,
        "visits": int(getattr(node, "N", 0)),
        "value_sum": float(getattr(node, "W", 0.0)),
        "q": float(getattr(node, "Q", 0.0)),
        "prior": float(getattr(node, "P", 0.0)),
        "bounds": _paper_bounds(paper),
        "paper_mask": mask.tolist(),
        "target_mask": target_mask.tolist(),
        "nn": {
            "start_map": start_map.tolist(),
            "end_map": end_map.tolist(),
            "value_estimate": value_estimate,
        },
        "checkpoint": _STATE.loaded_checkpoint,
    }


def _make_child(parent: Any, folded_paper: Any) -> Any:
    parent_cls = type(parent)
    parent_target = _target_mask(parent)
    parent_model = getattr(parent, "model", None)
    if parent_model is None:
        parent_model = _ensure_network()

    try:
        child = parent_cls(paper=folded_paper, model=parent_model, parent=parent, target_mask=parent_target)
    except TypeError:
        try:
            child = parent_cls(folded_paper, parent)
        except TypeError:
            child = _RuntimeNode(paper=folded_paper, parent=parent)

    if not hasattr(child, "children") or getattr(child, "children") is None:
        child.children = []

    child.parent = parent
    child.paper = folded_paper
    child.target_mask = np.array(parent_target, dtype=np.float32)
    if hasattr(parent, "model"):
        child.model = getattr(parent, "model")

    child.P = float(getattr(child, "P", 0.0))
    child.N = int(getattr(child, "N", 0))
    child.W = float(getattr(child, "W", 0.0))
    child.Q = float(getattr(child, "Q", 0.0))

    return child


def _backprop_after_manual_fold(parent: Any, child: Any, value_estimate: float) -> None:
    if child.N <= 0:
        child.N = 1
    child.W += value_estimate
    child.Q = child.W / child.N

    cursor = parent
    while cursor is not None:
        cursor.N = int(getattr(cursor, "N", 0)) + 1
        cursor.W = float(getattr(cursor, "W", 0.0)) + value_estimate
        cursor.Q = cursor.W / cursor.N if cursor.N > 0 else 0.0
        cursor = getattr(cursor, "parent", None)


def _list_checkpoints() -> list[dict[str, Any]]:
    checkpoint_path = _resolve_fixed_checkpoint()
    return [
        {
            "name": checkpoint_path.name,
            "path": str(checkpoint_path.resolve()),
            "mtime": float(checkpoint_path.stat().st_mtime),
            "loaded": str(checkpoint_path.resolve()) == _STATE.loaded_checkpoint,
        }
    ]


def _resolve_fixed_checkpoint() -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (CHECKPOINT_DIR / FIXED_CHECKPOINT_NAME).resolve()
    if checkpoint_path.exists():
        return checkpoint_path

    root_checkpoint = FIXED_CHECKPOINT_ROOT_PATH.resolve()
    if root_checkpoint.exists():
        shutil.copy2(root_checkpoint, checkpoint_path)
        return checkpoint_path

    raise FileNotFoundError(
        f"Fixed checkpoint '{FIXED_CHECKPOINT_NAME}' not found in '{CHECKPOINT_DIR}' "
        f"or repo root '{FIXED_CHECKPOINT_ROOT_PATH.parent}'."
    )


def _resolve_checkpoint(payload: dict[str, Any]) -> Path:
    _ = payload
    return _resolve_fixed_checkpoint()


def _load_checkpoint(path: Path) -> None:
    net = _ensure_network()
    device = next(net.parameters()).device

    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format.")

    net.load_state_dict(state_dict, strict=True)
    net.eval()
    _STATE.loaded_checkpoint = str(path.resolve())


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.get("/api/tree")
def api_tree() -> Any:
    tree, _ = _to_tree_view(_current_root())
    return jsonify(tree)


@app.get("/api/node/<path:node_id>")
def api_node(node_id: str) -> Any:
    node = _resolve_node(node_id)
    if node is None:
        return jsonify({"error": f"Node '{node_id}' not found"}), 404
    return jsonify(_node_payload(node, node_id))


@app.get("/api/checkpoints")
def api_checkpoints() -> Any:
    return jsonify(
        {
            "checkpoints": _list_checkpoints(),
            "loaded_checkpoint": _STATE.loaded_checkpoint,
            "device": str(_STATE.device) if _STATE.device is not None else str(_get_device()),
        }
    )


@app.post("/api/checkpoints/load")
def api_load_checkpoint() -> Any:
    payload = request.get_json(silent=True) or {}
    try:
        path = _resolve_checkpoint(payload)
        _load_checkpoint(path)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify(
        {
            "ok": True,
            "loaded_checkpoint": _STATE.loaded_checkpoint,
            "checkpoints": _list_checkpoints(),
        }
    )


@app.post("/api/fold")
def api_fold() -> Any:
    payload = request.get_json(silent=True) or {}
    node_id = str(payload.get("node_id", "root"))

    node = _resolve_node(node_id)
    if node is None:
        return jsonify({"ok": False, "error": f"Node '{node_id}' not found"}), 404

    paper = getattr(node, "paper", None)
    if paper is None:
        return jsonify({"ok": False, "error": "Selected node has no paper state"}), 400

    try:
        x1 = float(payload["x1"])
        y1 = float(payload["y1"])
        x2 = float(payload["x2"])
        y2 = float(payload["y2"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"ok": False, "error": "Fold requires numeric x1, y1, x2, y2"}), 400

    segment = Segment(Vec2(x1, y1), Vec2(x2, y2))
    folded_paper = paper.copy()
    if not folded_paper.fold(segment):
        return jsonify({"ok": False, "error": "Fold was invalid for this paper state"}), 400

    if not hasattr(node, "children") or getattr(node, "children") is None:
        node.children = []

    child = _make_child(node, folded_paper)
    node.children.append(child)

    mask = _paper_mask(folded_paper)
    _, _, value_estimate = _nn_outputs(mask, _target_mask(node))
    _backprop_after_manual_fold(node, child, value_estimate)

    tree, lookup = _to_tree_view(_current_root())
    new_node_id = None
    for candidate_id, candidate_node in lookup.items():
        if candidate_node is child:
            new_node_id = candidate_id
            break

    return jsonify({"ok": True, "new_node_id": new_node_id, "tree": tree})


@app.post("/api/expand")
def api_expand() -> Any:
    payload = request.get_json(silent=True) or {}
    node_id = str(payload.get("node_id", "root"))

    node = _resolve_node(node_id)
    if node is None:
        return jsonify({"ok": False, "error": f"Node '{node_id}' not found"}), 404

    expand_fn = getattr(node, "expand", None)
    if not callable(expand_fn):
        return jsonify({"ok": False, "error": "Selected node does not implement expand()"}), 400

    try:
        _ = _ensure_network()
        expand_fn()
    except Exception as exc:
        return jsonify({"ok": False, "error": f"expand() failed: {exc}"}), 400

    tree, _ = _to_tree_view(_current_root())
    return jsonify({"ok": True, "tree": tree, "selected_node_id": node_id})


@app.post("/api/select")
def api_select() -> Any:
    payload = request.get_json(silent=True) or {}
    node_id = str(payload.get("node_id", "root"))

    node = _resolve_node(node_id)
    if node is None:
        return jsonify({"ok": False, "error": f"Node '{node_id}' not found"}), 404

    select_fn = getattr(node, "select", None)
    if not callable(select_fn):
        return jsonify({"ok": False, "error": "Selected node does not implement select()"}), 400

    try:
        selected = select_fn()
    except Exception as exc:
        return jsonify({"ok": False, "error": f"select() failed: {exc}"}), 400

    if selected is None:
        tree, _ = _to_tree_view(_current_root())
        return jsonify(
            {"ok": False, "error": "select() returned no child", "tree": tree, "selected_node_id": node_id}
        ), 400

    if int(getattr(selected, "N", 0)) <= 0:
        selected.N = 1
        selected.W = float(getattr(selected, "W", 0.0))
        selected.Q = selected.W / selected.N

    tree, lookup = _to_tree_view(_current_root())
    selected_node_id = None
    for candidate_id, candidate_node in lookup.items():
        if candidate_node is selected:
            selected_node_id = candidate_id
            break

    return jsonify(
        {
            "ok": True,
            "tree": tree,
            "selected_node_id": selected_node_id,
            "selected_visible": selected_node_id is not None,
        }
    )


def _build_demo_tree() -> _DemoNode:
    root_paper = Paper()
    root = _DemoNode(N=14, W=7.2, Q=0.514, P=1.0, paper=root_paper)

    a_paper = root_paper.copy()
    a_paper.fold(Segment(Vec2(0.1, 0.1), Vec2(0.9, 0.8)))
    b_paper = root_paper.copy()
    b_paper.fold(Segment(Vec2(0.05, 0.8), Vec2(0.95, 0.15)))

    a = _DemoNode(N=8, W=5.6, Q=0.7, P=0.54, paper=a_paper, parent=root)
    b = _DemoNode(N=5, W=1.4, Q=0.28, P=0.31, paper=b_paper, parent=root)
    c = _DemoNode(N=0, W=0.0, Q=0.0, P=0.15, paper=root_paper.copy(), parent=root)

    a1_paper = a_paper.copy()
    a1_paper.fold(Segment(Vec2(0.0, 0.45), Vec2(1.0, 0.55)))
    a2_paper = a_paper.copy()
    a2_paper.fold(Segment(Vec2(0.5, 0.0), Vec2(0.5, 1.0)))

    b1_paper = b_paper.copy()
    b1_paper.fold(Segment(Vec2(0.0, 0.3), Vec2(1.0, 0.7)))

    a1 = _DemoNode(N=4, W=3.2, Q=0.8, P=0.6, paper=a1_paper, parent=a)
    a2 = _DemoNode(N=2, W=1.0, Q=0.5, P=0.25, paper=a2_paper, parent=a)
    a3 = _DemoNode(N=0, W=0.0, Q=0.0, P=0.15, paper=a_paper.copy(), parent=a)

    b1 = _DemoNode(N=2, W=0.6, Q=0.3, P=0.58, paper=b1_paper, parent=b)
    b2 = _DemoNode(N=0, W=0.0, Q=0.0, P=0.42, paper=b_paper.copy(), parent=b)

    root.children = [a, b, c]
    a.children = [a1, a2, a3]
    b.children = [b1, b2]
    return root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive MCTS visualizer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--demo", action="store_true", help="Populate with demo nodes")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.demo:
        set_tree_root(_build_demo_tree())

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
