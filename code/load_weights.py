"""Example: load released PGFA weights without releasing any dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

try:
    from .pgfa_full import load_graph_npz, load_pretrained, make_demo_graph, policy_from_logits
except ImportError:  # direct execution: python code/load_weights.py
    from pgfa_full import load_graph_npz, load_pretrained, make_demo_graph, policy_from_logits


def parse_args():
    package_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Load released PGFA checkpoint and run one demo forward pass.")
    parser.add_argument("--city", choices=["CD", "NY"], default="CD")
    parser.add_argument("--availability", choices=["100", "66", "33", "1.0", "0.66", "0.33"], default="33")
    parser.add_argument("--weights-root", default=str(package_root / "weights"))
    parser.add_argument("--graph-npz", default=None, help="Optional NPZ with action_targets, valid_mask, adjacency.")
    parser.add_argument("--features-npy", default=None, help="Optional node feature matrix [N, F] or batch [B, N, F].")
    parser.add_argument("--nodes", type=int, default=232, help="Demo node count when --graph-npz is not supplied.")
    return parser.parse_args()


def main():
    args = parse_args()
    graph = load_graph_npz(args.graph_npz) if args.graph_npz else make_demo_graph(args.nodes)
    model, feature_dim, prefix = load_pretrained(
        city=args.city,
        availability=args.availability,
        weights_root=args.weights_root,
        graph=graph,
    )

    n_nodes = graph["action_targets"].shape[0]
    if args.features_npy:
        features = np.load(args.features_npy).astype(np.float32)
        if features.ndim == 2:
            features = features[None, :, :]
    else:
        # Placeholder only.  Replace with your own enhanced regional features.
        features = np.zeros((1, n_nodes, feature_dim), dtype=np.float32)

    if features.shape[-1] != feature_dim:
        raise ValueError(f"{args.city} checkpoint expects feature_dim={feature_dim}, got {features.shape[-1]}")
    if features.shape[1] != n_nodes:
        raise ValueError(f"Feature node count {features.shape[1]} does not match graph node count {n_nodes}")

    logits, value, psi = model(tf.convert_to_tensor(features), training=False)
    probs = policy_from_logits(logits).numpy()
    actions = np.argmax(probs, axis=-1)

    print(f"Loaded checkpoint: {prefix}")
    print(f"city={args.city} availability={args.availability} nodes={n_nodes} feature_dim={feature_dim}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"value shape: {tuple(value.shape)} value[0]={float(value.numpy()[0]):.6f}")
    print(f"psi shape: {tuple(psi.shape)}")
    print(f"actions shape: {tuple(actions.shape)} first 10 actions: {actions[0, :10].tolist()}")


if __name__ == "__main__":
    main()
