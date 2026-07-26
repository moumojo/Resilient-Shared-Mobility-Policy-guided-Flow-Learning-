"""Discover, load, verify, and run every released paper checkpoint."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

try:
    from .pgfa_full import CITY_CONFIGS, build_model, load_graph_npz, make_demo_graph
except ImportError:  # direct execution: python code/run_test.py
    from pgfa_full import CITY_CONFIGS, build_model, load_graph_npz, make_demo_graph


BASELINE_VARIANTS = {
    "StructuralHeuristic_GNN": "structural_heuristic",
    "EdgeAware_GAT": "edge_aware_gat",
    "MultiModal_STGCN": "multimodal_stgcn",
    "WeightedLinearAttention_STGCN": "wla_stgcn",
    "MM_STMAP": "mm_stmap",
    "DigitalTwin_D3QN": "dt_d3qn",
    "QoSQueue_DTSM": "qos_queue_dtsm",
}

ABLATIONS = {
    "full",
    "no_enhanced_state",
    "no_policy_flow_prior",
    "no_policy_guided_message",
    "no_flow_aware_attention",
}


@dataclass(frozen=True)
class CheckpointSpec:
    checkpoint_id: str
    prefix: Path
    family: str
    variant: str
    city: str
    availability: str
    ablation: str

    def public_dict(self) -> Dict[str, str]:
        row = asdict(self)
        row["prefix"] = str(self.prefix)
        return row


def _infer_city(parts: Iterable[str]) -> str:
    lowered = [part.lower() for part in parts]
    if any(part == "ny" or "ablation_ny" in part or "ny_main" in part for part in lowered):
        return "NY"
    if any(part == "cd" or "ablation_cd" in part or "cd_ablation" in part for part in lowered):
        return "CD"
    if any("nyc" in part for part in lowered):
        return "NY"
    raise ValueError("Cannot infer city from checkpoint path")


def _infer_availability(parts: Iterable[str]) -> str:
    for part in parts:
        match = re.search(r"(?:avail_|(?:nyc_)?)(100|66|33)(?:_|$)", part.lower())
        if match:
            return match.group(1)
    raise ValueError("Cannot infer availability from checkpoint path")


def _infer_ablation(parts: Iterable[str]) -> str:
    lowered = {part.lower() for part in parts}
    for ablation in ABLATIONS:
        if ablation in lowered:
            return ablation
    return "full"


def _infer_variant(parts: Iterable[str]) -> Optional[str]:
    for part in parts:
        if part in BASELINE_VARIANTS:
            return BASELINE_VARIANTS[part]
    return None


def discover_checkpoints(result_root: Path) -> List[CheckpointSpec]:
    result_root = result_root.resolve()
    if not result_root.is_dir():
        raise FileNotFoundError(f"Result directory does not exist: {result_root}")

    specs = []
    for index_path in sorted(result_root.rglob("best_pgfa_rl_weights.index")):
        relative_dir = index_path.parent.relative_to(result_root)
        parts = relative_dir.parts
        variant = _infer_variant(parts)
        specs.append(
            CheckpointSpec(
                checkpoint_id=relative_dir.as_posix(),
                prefix=index_path.with_suffix(""),
                family="baseline" if variant else "pgfa",
                variant=variant or "pgfa",
                city=_infer_city(parts),
                availability=_infer_availability(parts),
                ablation=_infer_ablation(parts),
            )
        )
    return specs


def resolve_checkpoint(specs: List[CheckpointSpec], query: str) -> CheckpointSpec:
    normalized = query.replace("\\", "/").rstrip("/")
    exact = [spec for spec in specs if spec.checkpoint_id == normalized]
    if len(exact) == 1:
        return exact[0]
    matches = [spec for spec in specs if normalized in spec.checkpoint_id]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"No checkpoint matches {query!r}; use --list to inspect checkpoint IDs")
    choices = "\n  ".join(spec.checkpoint_id for spec in matches[:20])
    raise ValueError(f"Checkpoint query {query!r} is ambiguous:\n  {choices}")


def build_and_load(spec: CheckpointSpec, graph: Dict[str, np.ndarray]):
    import tensorflow as tf

    feature_dim = CITY_CONFIGS[spec.city]["feature_dim"]
    n_nodes = int(graph["action_targets"].shape[0])
    if spec.family == "baseline":
        try:
            from .baseline_models import build_model_class
        except ImportError:  # direct execution: python code/run_test.py
            from baseline_models import build_model_class

        model_class = build_model_class(spec.variant)
        model = model_class(graph, hidden_dim=CITY_CONFIGS[spec.city]["hidden_dim"])
        _ = model(tf.zeros((1, n_nodes, feature_dim), dtype=tf.float32), training=False)
    else:
        model, feature_dim = build_model(spec.city, graph=graph, ablation=spec.ablation)

    status = model.load_weights(str(spec.prefix))
    try:
        status.assert_existing_objects_matched()
        status.expect_partial()
    except AttributeError:
        pass
    return model, feature_dim


def infer_checkpoint_nodes(spec: CheckpointSpec, fallback: int = 232) -> int:
    """Infer node count from node-specific checkpoint variables when present."""
    import tensorflow as tf

    for variable_name, shape in tf.train.list_variables(str(spec.prefix)):
        if "learnable_spatial_alpha" in variable_name and len(shape) == 2 and shape[1] == 7:
            return int(shape[0])
    return int(fallback)


def batched_predict(model, features: np.ndarray, batch_size: int):
    import tensorflow as tf

    logits_rows = []
    value_rows = []
    aux_rows = []
    for start in range(0, features.shape[0], batch_size):
        batch = tf.convert_to_tensor(features[start : start + batch_size], dtype=tf.float32)
        logits, value, aux = model(batch, training=False)
        logits_rows.append(logits.numpy())
        value_rows.append(value.numpy())
        aux_rows.append(aux.numpy())
    logits = np.concatenate(logits_rows, axis=0)
    values = np.concatenate(value_rows, axis=0)
    aux = np.concatenate(aux_rows, axis=0)
    probabilities = tf.nn.softmax(logits, axis=-1).numpy()
    actions = np.argmax(probabilities, axis=-1).astype(np.int32)
    return logits, probabilities, actions, values, aux


def verify_all(specs: List[CheckpointSpec], graph_path: Optional[str], nodes: Optional[int]) -> int:
    import tensorflow as tf

    failures = []
    graph_cache = {}
    for position, spec in enumerate(specs, start=1):
        try:
            checkpoint_nodes = int(nodes) if nodes is not None else infer_checkpoint_nodes(spec)
            cache_key = (spec.city, checkpoint_nodes)
            if cache_key not in graph_cache:
                graph_cache[cache_key] = load_graph_npz(graph_path) if graph_path else make_demo_graph(checkpoint_nodes)
            tf.keras.backend.clear_session()
            model, feature_dim = build_and_load(spec, graph_cache[cache_key])
            n_nodes = graph_cache[cache_key]["action_targets"].shape[0]
            output = model(tf.zeros((1, n_nodes, feature_dim), dtype=tf.float32), training=False)
            if len(output) != 3:
                raise ValueError(f"Expected three model outputs, received {len(output)}")
            print(f"[{position:03d}/{len(specs):03d}] OK {spec.checkpoint_id}")
        except Exception as exc:  # continue to provide a complete checkpoint audit
            failures.append((spec.checkpoint_id, f"{type(exc).__name__}: {exc}"))
            print(f"[{position:03d}/{len(specs):03d}] FAIL {spec.checkpoint_id}: {exc}")

    print(json.dumps({"total": len(specs), "passed": len(specs) - len(failures), "failed": len(failures)}, indent=2))
    if failures:
        print("Failed checkpoints:")
        for checkpoint_id, error in failures:
            print(f"  {checkpoint_id}: {error}")
        return 1
    return 0


def parse_args():
    package_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default=str(package_root / "result"))
    parser.add_argument("--list", action="store_true", help="List every discovered checkpoint and exit.")
    parser.add_argument("--verify-all", action="store_true", help="Build, load, and forward-test every checkpoint.")
    parser.add_argument("--checkpoint", help="Checkpoint ID from --list; a unique substring is also accepted.")
    parser.add_argument("--graph-npz", help="Graph arrays: action_targets, valid_mask, and adjacency.")
    parser.add_argument("--features-npy", help="Features with shape [T,N,F] or [N,F].")
    parser.add_argument("--output-dir", default="test_outputs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--nodes",
        type=int,
        default=None,
        help="Override node count for checkpoint-only verification; otherwise infer it from the checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    specs = discover_checkpoints(Path(args.result_root))
    if not specs:
        raise SystemExit(f"No checkpoints found under {args.result_root}")

    if args.list:
        for spec in specs:
            print(
                f"{spec.checkpoint_id}\t{spec.family}\t{spec.variant}\t"
                f"{spec.city}\tavail_{spec.availability}\t{spec.ablation}"
            )
        print(f"Total checkpoints: {len(specs)}")
        return 0

    if args.verify_all:
        return verify_all(specs, args.graph_npz, args.nodes)

    if not args.checkpoint:
        raise SystemExit("Choose --list, --verify-all, or --checkpoint CHECKPOINT_ID")

    spec = resolve_checkpoint(specs, args.checkpoint)
    checkpoint_nodes = args.nodes if args.nodes is not None else infer_checkpoint_nodes(spec)
    graph = load_graph_npz(args.graph_npz) if args.graph_npz else make_demo_graph(checkpoint_nodes)
    model, feature_dim = build_and_load(spec, graph)
    n_nodes = int(graph["action_targets"].shape[0])

    if args.features_npy:
        features = np.load(args.features_npy).astype(np.float32)
        if features.ndim == 2:
            features = features[None, :, :]
    else:
        features = np.zeros((1, n_nodes, feature_dim), dtype=np.float32)

    if features.ndim != 3:
        raise ValueError("Features must have shape [T,N,F] or [N,F]")
    if features.shape[1] != n_nodes:
        raise ValueError(f"Feature node count {features.shape[1]} does not match graph node count {n_nodes}")
    if features.shape[2] != feature_dim:
        raise ValueError(f"{spec.city} checkpoint expects feature_dim={feature_dim}, got {features.shape[2]}")

    logits, probabilities, actions, values, aux = batched_predict(model, features, args.batch_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "logits.npy", logits)
    np.save(output_dir / "action_probs.npy", probabilities)
    np.save(output_dir / "actions.npy", actions)
    np.save(output_dir / "values.npy", values)
    np.save(output_dir / "aux.npy", aux)
    summary = {
        **spec.public_dict(),
        "num_snapshots": int(features.shape[0]),
        "num_nodes": int(features.shape[1]),
        "feature_dim": int(features.shape[2]),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
