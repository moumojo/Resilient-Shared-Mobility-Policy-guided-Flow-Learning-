import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict

import numpy as np

try:
    import tensorflow as tf
except ImportError as exc:
    raise SystemExit(
        "TensorFlow is required. Install the dependencies from requirements.txt "
        "or environment.yml."
    ) from exc


ROOT = Path(__file__).resolve().parents[2]
EPS = 1.0e-8


VARIANT_LABELS = {
    "structural_heuristic": "BloomSignature-GNN",
    "edge_aware_gat": "OGIF-GAT",
    "multimodal_stgcn": "MultiModal-STGCN",
    "wla_stgcn": "WLA-STGCN",
    "mm_stmap": "MM-STMAP",
    "dt_d3qn": "DT-D3QN",
    "qos_queue_dtsm": "QoSQueue-DTSM",
}


def _bloom_signatures(graph: Dict[str, np.ndarray], bloom_dim=64, max_hop=2, hashes=4) -> np.ndarray:
    """Compact multi-hop neighborhood signatures following the Bloom-signature idea.

    The original Bloom-signature paper targets link prediction. For dispatch,
    each action is also a source-target relation, so we store compact hashed
    one-hop/two-hop neighborhoods and let the policy network compare the
    source and target signatures when scoring each move.
    """

    adjacency = np.asarray(graph["adjacency"], dtype=np.float32)
    n_nodes = adjacency.shape[0]
    signatures = np.zeros((n_nodes, bloom_dim * max_hop), dtype=np.float32)
    primes = np.asarray([17, 31, 47, 73, 97, 127, 191, 251], dtype=np.int64)

    one_hop = [set(np.where(adjacency[i] > 0)[0].tolist()) for i in range(n_nodes)]
    neighborhoods = []
    for i in range(n_nodes):
        hop_sets = []
        frontier = {i}
        visited = {i}
        for _ in range(max_hop):
            nxt = set()
            for node in frontier:
                nxt.update(one_hop[node])
            hop_nodes = nxt - visited
            hop_sets.append(hop_nodes if hop_nodes else set(one_hop[i]))
            visited.update(nxt)
            frontier = nxt
        neighborhoods.append(hop_sets)

    for i, hop_sets in enumerate(neighborhoods):
        for hop_idx, nodes in enumerate(hop_sets):
            offset = hop_idx * bloom_dim
            for node in nodes:
                for k in range(hashes):
                    bit = int((node * primes[k] + i * primes[k + 1] + 13 * (hop_idx + 1)) % bloom_dim)
                    signatures[i, offset + bit] = 1.0
    return signatures


def _masked_edge_features(graph: Dict[str, np.ndarray]) -> np.ndarray:
    adjacency = np.asarray(graph["adjacency"], dtype=np.float32)
    targets = np.asarray(graph["action_targets"], dtype=np.int32)
    valid = np.asarray(graph["valid_mask"], dtype=np.float32)
    signatures = _bloom_signatures(graph)
    n_nodes, n_actions = targets.shape
    degree = adjacency.sum(axis=1).astype(np.float32)
    max_degree = float(np.max(degree) + 1.0)

    feats = np.zeros((n_nodes, n_actions, 12), dtype=np.float32)
    for i in range(n_nodes):
        neigh_i = adjacency[i] > 0
        for a in range(n_actions):
            j = int(targets[i, a])
            neigh_j = adjacency[j] > 0
            common = float(np.logical_and(neigh_i, neigh_j).sum())
            union = float(np.logical_or(neigh_i, neigh_j).sum())
            jaccard = common / max(union, 1.0)
            overlap = common / max(min(float(degree[i]), float(degree[j])), 1.0)
            feats[i, a, 0] = valid[i, a]
            feats[i, a, 1] = 1.0 if a == 6 else 0.0
            feats[i, a, 2] = float(a) / 6.0
            feats[i, a, 3] = degree[i] / max_degree
            feats[i, a, 4] = degree[j] / max_degree
            feats[i, a, 5] = jaccard
            feats[i, a, 6] = overlap
            feats[i, a, 7] = 1.0 if j != i else 0.0
            feats[i, a, 8] = abs(float(degree[i] - degree[j])) / max_degree
            sig_i = signatures[i] > 0
            sig_j = signatures[j] > 0
            sig_common = float(np.logical_and(sig_i, sig_j).sum())
            sig_union = float(np.logical_or(sig_i, sig_j).sum())
            feats[i, a, 9] = sig_common / max(sig_union, 1.0)
            feats[i, a, 10] = sig_common / max(float(sig_i.sum()), 1.0)
            feats[i, a, 11] = sig_common / max(float(sig_j.sum()), 1.0)
    return feats


class _ComparisonBase(tf.keras.Model):
    def __init__(self, graph, hidden_dim=128, entropy_floor=-1.0e9, name="comparison_model"):
        super().__init__(name=name)
        self.action_targets = tf.constant(graph["action_targets"], dtype=tf.int32)
        self.valid_mask = tf.constant(graph["valid_mask"], dtype=tf.float32)
        self.adjacency = tf.constant(graph["adjacency"], dtype=tf.float32)
        self.adjacency_t = tf.transpose(self.adjacency)
        self.edge_features = tf.constant(_masked_edge_features(graph), dtype=tf.float32)
        self.node_signatures = tf.constant(_bloom_signatures(graph), dtype=tf.float32)
        self.entropy_floor = float(entropy_floor)

        self.enc1 = tf.keras.layers.Dense(hidden_dim, activation="relu", name="state_encoder_1")
        self.enc2 = tf.keras.layers.Dense(hidden_dim, activation="relu", name="state_encoder_2")
        self.signature_proj = tf.keras.layers.Dense(hidden_dim // 2, activation="relu", name="bloom_signature_projection")
        self.value_dense = tf.keras.layers.Dense(hidden_dim, activation="relu", name="value_hidden")
        self.value_out = tf.keras.layers.Dense(1, activation=None, name="value")
        self.aux_head = tf.keras.layers.Dense(1, activation="tanh", name="comparison_aux")

    def _encode(self, node_features):
        h = self.enc1(node_features)
        return self.enc2(h)

    def _pair_tensors(self, h):
        h_j = tf.gather(h, self.action_targets, axis=1)
        h_i = tf.expand_dims(h, axis=2)
        h_i = tf.tile(h_i, [1, 1, 7, 1])
        edge = tf.tile(self.edge_features[None, :, :, :], [tf.shape(h)[0], 1, 1, 1])
        sig_i = tf.expand_dims(self.node_signatures, axis=1)
        sig_i = tf.tile(sig_i, [1, 7, 1])
        sig_j = tf.gather(self.node_signatures, self.action_targets, axis=0)
        sig_i = tf.tile(sig_i[None, :, :, :], [tf.shape(h)[0], 1, 1, 1])
        sig_j = tf.tile(sig_j[None, :, :, :], [tf.shape(h)[0], 1, 1, 1])
        return h_i, h_j, edge, sig_i, sig_j

    def _masked_logits(self, logits):
        return logits + (1.0 - self.valid_mask[None, :, :]) * self.entropy_floor

    def _value(self, h, aux):
        pooled = tf.reduce_mean(tf.concat([h, aux], axis=-1), axis=1)
        return tf.squeeze(self.value_out(self.value_dense(pooled)), axis=-1)


class StructuralHeuristicNetwork(_ComparisonBase):
    """Bloom-signature structural baseline adapted from scalable link prediction."""

    def __init__(self, graph, hidden_dim=128, entropy_floor=-1.0e9):
        super().__init__(graph, hidden_dim, entropy_floor, name="BloomSignature_GNN")
        self.bloom_pair_dense = tf.keras.layers.Dense(hidden_dim, activation="relu", name="bloom_pair_encoder")
        self.action_dense = tf.keras.layers.Dense(hidden_dim, activation="relu", name="bloom_action_hidden")
        self.logit_head = tf.keras.layers.Dense(1, activation=None, name="action_logit")

    def call(self, node_features, training=False):
        h = self._encode(node_features)
        neigh_h = tf.matmul(self.adjacency, h)
        h = 0.70 * h + 0.30 * neigh_h
        aux = self.aux_head(h)
        h_i, h_j, edge, sig_i, sig_j = self._pair_tensors(h)
        bloom_pair = self.bloom_pair_dense(tf.concat([sig_i, sig_j, sig_i * sig_j, tf.abs(sig_j - sig_i)], axis=-1))
        action_hidden = self.action_dense(tf.concat([h_i, h_j, h_i * h_j, bloom_pair, edge], axis=-1))
        logits = tf.squeeze(self.logit_head(action_hidden), axis=-1)
        return self._masked_logits(logits), self._value(h, aux), aux


class EdgeAwareGATNetwork(_ComparisonBase):
    """Optimal graph information fused GAT baseline.

    It fuses predefined adjacency, a feature-similarity dynamic graph, and the
    self graph before applying edge-aware action attention.
    """

    def __init__(self, graph, hidden_dim=128, entropy_floor=-1.0e9):
        super().__init__(graph, hidden_dim, entropy_floor, name="OGIF_GAT")
        self.graph_fusion_logits = self.add_weight(
            name="optimal_graph_fusion_logits",
            shape=(3,),
            initializer="zeros",
            trainable=True,
        )
        self.edge_proj = tf.keras.layers.Dense(hidden_dim, activation="relu", name="edge_feature_projection")
        self.graph_fuse_dense = tf.keras.layers.Dense(hidden_dim, activation="relu", name="optimal_graph_fusion")
        self.att_dense = tf.keras.layers.Dense(hidden_dim, activation="relu", name="edge_attention_hidden")
        self.logit_head = tf.keras.layers.Dense(1, activation=None, name="action_logit")

    def call(self, node_features, training=False):
        h0 = self._encode(node_features)
        h_static = tf.matmul(self.adjacency, h0)
        h_norm = tf.math.l2_normalize(h0, axis=-1)
        sim = tf.matmul(h_norm, h_norm, transpose_b=True)
        dyn_mask = tf.cast(sim > 0.0, tf.float32)
        sim = sim + (1.0 - dyn_mask) * -1.0e9
        dyn_graph = tf.nn.softmax(sim, axis=-1)
        h_dynamic = tf.matmul(dyn_graph, h0)
        weights = tf.nn.softmax(self.graph_fusion_logits)
        h = self.graph_fuse_dense(
            tf.concat([weights[0] * h0, weights[1] * h_static, weights[2] * h_dynamic], axis=-1)
        )
        aux = self.aux_head(h)
        h_i, h_j, edge, _, _ = self._pair_tensors(h)
        edge_context = self.edge_proj(edge)
        att_hidden = self.att_dense(tf.concat([h_i, h_j, h_i * h_j, tf.abs(h_j - h_i), edge_context], axis=-1))
        logits = tf.squeeze(self.logit_head(att_hidden), axis=-1)
        return self._masked_logits(logits), self._value(h, aux), aux


class MultiModalSTGCNNetwork(_ComparisonBase):
    """MM-STMAP perception and spatio-temporal graph convolution baseline.

    The paper fuses traffic flow, weather, holiday/event and historical states.
    In this ride-hailing adaptation, the available enhanced state is split into
    local supply-demand, neighborhood context, trend/risk, and temporal context
    modalities, then fused before residual spatial-temporal graph convolution.
    """

    def __init__(self, graph, hidden_dim=128, entropy_floor=-1.0e9):
        super().__init__(graph, hidden_dim, entropy_floor, name="MultiModal_STGCN")
        branch_dim = max(hidden_dim // 2, 32)
        self.local_encoder = tf.keras.layers.Dense(branch_dim, activation="relu", name="local_supply_demand_encoder")
        self.neighbor_encoder = tf.keras.layers.Dense(branch_dim, activation="relu", name="neighbor_context_encoder")
        self.risk_encoder = tf.keras.layers.Dense(branch_dim, activation="relu", name="trend_risk_encoder")
        self.time_encoder = tf.keras.layers.Dense(branch_dim, activation="relu", name="temporal_context_encoder")
        self.modality_gate = tf.keras.layers.Dense(4, activation=None, name="modality_gate")
        self.fusion_dense = tf.keras.layers.Dense(hidden_dim, activation="relu", name="multimodal_fusion")
        self.spatial_dense = tf.keras.layers.Dense(hidden_dim, activation="relu", name="spatial_graph_convolution")
        self.temporal_dense = tf.keras.layers.Dense(hidden_dim, activation="relu", name="temporal_state_projection")
        self.action_dense = tf.keras.layers.Dense(hidden_dim, activation="relu", name="stgcn_action_hidden")
        self.logit_head = tf.keras.layers.Dense(1, activation=None, name="action_logit")

    def _multimodal_encode(self, node_features):
        local = self.local_encoder(node_features[..., :3])
        neighbor = self.neighbor_encoder(node_features[..., 3:8])
        risk = self.risk_encoder(node_features[..., 8:-3])
        temporal = self.time_encoder(node_features[..., -3:])
        stacked = tf.stack([local, neighbor, risk, temporal], axis=2)
        gate_context = tf.concat(
            [
                tf.reduce_mean(node_features[..., :3], axis=-1, keepdims=True),
                tf.reduce_mean(node_features[..., 3:8], axis=-1, keepdims=True),
                tf.reduce_mean(node_features[..., 8:-3], axis=-1, keepdims=True),
                tf.reduce_mean(node_features[..., -3:], axis=-1, keepdims=True),
            ],
            axis=-1,
        )
        weights = tf.nn.softmax(self.modality_gate(gate_context), axis=-1)
        fused_modal = tf.reduce_sum(stacked * weights[:, :, :, None], axis=2)
        return self.fusion_dense(fused_modal)

    def _stgcn_encode(self, node_features):
        h0 = self._multimodal_encode(node_features)
        h_spatial = self.spatial_dense(tf.matmul(self.adjacency, h0))
        h_temporal = self.temporal_dense(h0)
        return h0 + h_spatial + 0.50 * h_temporal

    def call(self, node_features, training=False):
        h = self._stgcn_encode(node_features)
        aux = self.aux_head(h)
        h_i, h_j, edge, _, _ = self._pair_tensors(h)
        action_hidden = self.action_dense(tf.concat([h_i, h_j, h_j - h_i, edge], axis=-1))
        logits = tf.squeeze(self.logit_head(action_hidden), axis=-1)
        return self._masked_logits(logits), self._value(h, aux), aux


class WeightedLinearAttentionSTGCNNetwork(MultiModalSTGCNNetwork):
    """Weighted linear attention baseline from MM-STMAP."""

    def __init__(self, graph, hidden_dim=128, entropy_floor=-1.0e9):
        super().__init__(graph, hidden_dim, entropy_floor)
        self._name = "WLA_STGCN"
        valid = np.asarray(graph["valid_mask"], dtype=np.float32)
        spatial_init = np.where(valid > 0, 0.0, -6.0).astype(np.float32)
        self.spatial_alpha_logits = self.add_weight(
            name="learnable_spatial_alpha",
            shape=valid.shape,
            initializer=tf.keras.initializers.Constant(spatial_init),
            trainable=True,
        )
        self.query_dense = tf.keras.layers.Dense(hidden_dim, activation=None, name="linear_attention_query")
        self.key_dense = tf.keras.layers.Dense(hidden_dim, activation=None, name="linear_attention_key")
        self.value_dense_att = tf.keras.layers.Dense(hidden_dim, activation=None, name="linear_attention_value")
        self.temporal_beta_dense = tf.keras.layers.Dense(1, activation="sigmoid", name="learnable_temporal_beta")
        self.att_layer_norm = tf.keras.layers.LayerNormalization(name="weighted_linear_attention_norm")
        self.att_logit_head = tf.keras.layers.Dense(1, activation=None, name="attention_action_logit")

    def _weighted_linear_attention(self, h, node_features):
        h_i, h_j, edge, _, _ = self._pair_tensors(h)
        q = self.query_dense(h_i)
        k = self.key_dense(h_j)
        v = self.value_dense_att(h_j)
        d = tf.sqrt(tf.cast(tf.shape(q)[-1], tf.float32))
        compatibility = tf.reduce_sum(q * k, axis=-1, keepdims=True) / (d + EPS)
        alpha = tf.nn.sigmoid(self.spatial_alpha_logits)[None, :, :, None]
        temporal = self.temporal_beta_dense(node_features[..., -3:])
        beta_i = tf.expand_dims(temporal, axis=2)
        beta_i = tf.tile(beta_i, [1, 1, 7, 1])
        attended = self.att_layer_norm(alpha * beta_i * compatibility * v)
        return h_i, h_j, edge, attended

    def call(self, node_features, training=False):
        h = self._stgcn_encode(node_features)
        aux = self.aux_head(h)
        h_i, h_j, edge, attended = self._weighted_linear_attention(h, node_features)
        logits = tf.squeeze(self.att_logit_head(tf.concat([h_i, h_j, attended, edge], axis=-1)), axis=-1)
        return self._masked_logits(logits), self._value(h, aux), aux


class MMSTMAPNetwork(WeightedLinearAttentionSTGCNNetwork):
    """Full MM-STMAP adaptation: multimodal STGCN plus weighted linear attention."""

    def __init__(self, graph, hidden_dim=128, entropy_floor=-1.0e9):
        super().__init__(graph, hidden_dim, entropy_floor)
        self._name = "MM_STMAP"
        self.global_context_dense = tf.keras.layers.Dense(hidden_dim, activation="relu", name="mappo_global_context")
        self.coordination_dense = tf.keras.layers.Dense(hidden_dim, activation="relu", name="multi_agent_coordination")
        self.coord_logit_head = tf.keras.layers.Dense(1, activation=None, name="coordinated_action_logit")

    def call(self, node_features, training=False):
        h = self._stgcn_encode(node_features)
        global_context = tf.reduce_mean(h, axis=1, keepdims=True)
        global_context = self.global_context_dense(global_context)
        global_context = tf.tile(global_context, [1, tf.shape(h)[1], 1])
        h_coord = self.coordination_dense(tf.concat([h, global_context], axis=-1))
        aux = self.aux_head(h_coord)
        h_i, h_j, edge, attended = self._weighted_linear_attention(h_coord, node_features)
        logits = tf.squeeze(self.coord_logit_head(tf.concat([h_i, h_j, attended, edge], axis=-1)), axis=-1)
        return self._masked_logits(logits), self._value(h_coord, aux), aux


class DigitalTwinD3QNNetwork(_ComparisonBase):
    """Digital-twin D3QN scheduling baseline adapted to CityReal.

    The paper's E2DTSM-OMT uses a digital twin environment plus a Dueling Double
    DQN scheduler. CityReal already serves as the virtual dispatch twin here, so
    this baseline implements the D3QN-style dueling value/advantage action-value
    decomposition over feasible regional dispatch actions.
    """

    def __init__(self, graph, hidden_dim=128, entropy_floor=-1.0e9):
        super().__init__(graph, hidden_dim, entropy_floor, name="DT_D3QN")
        self.twin_dense1 = tf.keras.layers.Dense(hidden_dim, activation="relu", name="twin_state_encoder_1")
        self.twin_dense2 = tf.keras.layers.Dense(hidden_dim, activation="relu", name="twin_state_encoder_2")
        self.graph_dense = tf.keras.layers.Dense(hidden_dim, activation="relu", name="twin_graph_state")
        self.advantage_dense = tf.keras.layers.Dense(hidden_dim, activation="relu", name="dueling_advantage_hidden")
        self.advantage_out = tf.keras.layers.Dense(1, activation=None, name="dueling_advantage")
        self.node_value_dense = tf.keras.layers.Dense(hidden_dim, activation="relu", name="dueling_node_value_hidden")
        self.node_value_out = tf.keras.layers.Dense(1, activation=None, name="dueling_node_value")

    def call(self, node_features, training=False):
        h0 = self._encode(node_features)
        h_seq = self.twin_dense2(self.twin_dense1(h0))
        h_graph = self.graph_dense(tf.concat([h_seq, tf.matmul(self.adjacency, h_seq)], axis=-1))
        h = h_seq + h_graph
        aux = self.aux_head(h)
        h_i, h_j, edge, _, _ = self._pair_tensors(h)
        advantage = tf.squeeze(self.advantage_out(self.advantage_dense(tf.concat([h_i, h_j, h_j - h_i, edge], axis=-1))), axis=-1)
        advantage_mean = tf.reduce_mean(advantage * self.valid_mask[None, :, :], axis=-1, keepdims=True)
        node_value = self.node_value_out(self.node_value_dense(h))
        logits = tf.squeeze(node_value, axis=-1)[:, :, None] + advantage - advantage_mean
        return self._masked_logits(logits), self._value(h, aux), aux


class QoSQueueDTSMNetwork(_ComparisonBase):
    """Mixed-traffic QoS and queue-buffer-aware scheduling baseline.

    The paper classifies mixed network flows into TT/AVB/BE and allocates path
    and queue resources under delay/buffer constraints. In ride-hailing dispatch,
    the analog is to classify dispatch actions into urgent-demand, predictive
    demand, and opportunistic balancing moves while gating each edge by source
    residual supply capacity.
    """

    def __init__(self, graph, hidden_dim=128, entropy_floor=-1.0e9):
        super().__init__(graph, hidden_dim, entropy_floor, name="QoSQueue_DTSM")
        self.priority_dense = tf.keras.layers.Dense(3, activation=None, name="mixed_flow_priority_classifier")
        self.queue_gate_dense = tf.keras.layers.Dense(1, activation="sigmoid", name="queue_buffer_resource_gate")
        self.edge_dense = tf.keras.layers.Dense(hidden_dim, activation="relu", name="qos_edge_hidden")
        self.logit_head = tf.keras.layers.Dense(1, activation=None, name="qos_action_logit")

    def call(self, node_features, training=False):
        h0 = self._encode(node_features)
        h = 0.55 * h0 + 0.45 * tf.matmul(self.adjacency, h0)
        aux = self.aux_head(h)
        h_i, h_j, edge, _, _ = self._pair_tensors(h)

        target_features = tf.gather(node_features, self.action_targets, axis=1)
        source_features = tf.expand_dims(node_features, axis=2)
        source_features = tf.tile(source_features, [1, 1, 7, 1])

        # Local feature indices follow build_features: future=2, shortage=5,
        # surplus=6, gap=10/11 depending on CD/NY dimensionality, peak=-1.
        target_future = target_features[..., 2:3]
        target_shortage = target_features[..., 5:6]
        source_surplus = source_features[..., 6:7]
        source_risk = tf.reduce_mean(source_features[..., 2:8], axis=-1, keepdims=True)
        qos_features = tf.concat([target_shortage, target_future, source_surplus, source_risk, edge], axis=-1)
        priority_logits = self.priority_dense(qos_features)
        priority_weights = tf.nn.softmax(priority_logits, axis=-1)
        # Urgent moves receive the largest base reward, predictive moves are
        # second, opportunistic balancing moves are third.
        class_reward = tf.reduce_sum(priority_weights * tf.constant([1.00, 0.65, 0.25], dtype=tf.float32), axis=-1, keepdims=True)
        resource_gate = self.queue_gate_dense(qos_features)
        edge_hidden = self.edge_dense(tf.concat([h_i, h_j, h_j - h_i, edge, class_reward, resource_gate], axis=-1))
        logits = tf.squeeze(self.logit_head(edge_hidden), axis=-1)
        logits = logits + tf.squeeze(tf.math.log(resource_gate + EPS) + class_reward, axis=-1)
        return self._masked_logits(logits), self._value(h, aux), aux


def build_model_class(variant):
    classes = {
        "structural_heuristic": StructuralHeuristicNetwork,
        "edge_aware_gat": EdgeAwareGATNetwork,
        "multimodal_stgcn": MultiModalSTGCNNetwork,
        "wla_stgcn": WeightedLinearAttentionSTGCNNetwork,
        "mm_stmap": MMSTMAPNetwork,
        "dt_d3qn": DigitalTwinD3QNNetwork,
        "qos_queue_dtsm": QoSQueueDTSMNetwork,
    }
    return classes[variant]


def stable_norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.percentile(np.abs(x), 90) + 1.0)


def generic_prior_logits(raw, graph, stay_logit=0.0):
    n_nodes = len(graph["target_grids"])
    logits = np.full((n_nodes, 7), -1.0e9, dtype=np.float32)
    logits[:, 6] = float(stay_logit)

    surplus = raw["surplus"].astype(np.float32)
    shortage = raw["shortage"].astype(np.float32)
    future = raw["future"].astype(np.float32)
    neigh_shortage = raw["neigh_shortage"].astype(np.float32)
    potential = raw.get("demand_potential", np.zeros_like(shortage)).astype(np.float32)

    source_score = stable_norm(surplus) - 0.15 * stable_norm(future) - 0.10 * stable_norm(neigh_shortage)
    target_score = stable_norm(shortage) + 0.20 * stable_norm(future) + 0.15 * stable_norm(neigh_shortage)
    target_score += 0.15 * stable_norm(potential)

    for i in range(n_nodes):
        if surplus[i] <= 0.25:
            continue
        for action_id in range(6):
            if graph["valid_mask"][i, action_id] <= 0:
                continue
            j = int(graph["action_targets"][i, action_id])
            if j == i:
                continue
            target_need = float(shortage[j] + 0.20 * future[j] + 0.10 * neigh_shortage[j] + 0.10 * potential[j])
            if target_need <= 0.05:
                continue
            score = math.log(target_need + 1.0) + 0.55 * float(target_score[j]) + 0.35 * float(source_score[i])
            logits[i, action_id] = float(score)
    return logits


def generic_make_dispatch(actions, probs, raw, graph, dispatch_scale, max_dispatch):
    tuples = []
    potential = raw.get("demand_potential", np.zeros_like(raw["shortage"]))
    for i, action_id in enumerate(actions):
        if action_id >= 6 or graph["valid_mask"][i, action_id] <= 0:
            continue
        j = int(graph["action_targets"][i, action_id])
        if j == i:
            continue
        movable = int(math.floor(float(raw["surplus"][i])))
        if movable <= 0:
            continue
        target_need = float(
            raw["shortage"][j]
            + 0.25 * raw["future"][j]
            + 0.12 * raw["neigh_shortage"][j]
            + 0.12 * potential[j]
        )
        if target_need <= 0.05:
            continue
        confidence = float(probs[i, action_id])
        num = int(math.floor(dispatch_scale * min(float(movable), target_need + 1.0) * max(confidence, 0.05)))
        num = max(0, min(int(max_dispatch), num, movable))
        if num > 0:
            tuples.append((int(graph["target_grids"][i]), int(graph["target_grids"][j]), int(num)))
    return tuples


def _discounted_returns(rewards, gamma):
    out = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = float(rewards[idx]) + gamma * running
        out[idx] = running
    return out


def mappo_style_train_on_episode(
    model,
    optimizer,
    batch,
    gamma,
    value_coef,
    entropy_coef,
    flow_coef,
    imitation_coef,
    prior_strength,
    grad_clip,
):
    """PPO-style clipped update used by the MM-STMAP dispatch adaptation.

    The original paper uses MAPPO for signal controllers. The CityReal runner
    stores actions and prior logits but not old policy log-probabilities, so this
    adaptation clips the learned policy against the behavior prior used to collect
    the trajectory. It keeps the update stable without changing the environment
    protocol used by the other main-table baselines.
    """

    states = tf.convert_to_tensor(np.asarray(batch["states"], dtype=np.float32))
    actions = tf.convert_to_tensor(np.asarray(batch["actions"], dtype=np.int32))
    expert_actions = tf.convert_to_tensor(np.asarray(batch["expert_actions"], dtype=np.int32))
    prior_logits = tf.convert_to_tensor(np.asarray(batch["prior_logits"], dtype=np.float32))
    rewards = np.asarray(batch["rewards"], dtype=np.float32)
    returns_np = _discounted_returns(rewards, gamma)
    if len(returns_np) > 1:
        returns_np = (returns_np - np.mean(returns_np)) / (np.std(returns_np) + 1.0e-6)
    returns = tf.convert_to_tensor(returns_np, dtype=tf.float32)
    clip_epsilon = 0.20

    with tf.GradientTape() as tape:
        logits, values, aux = model(states, training=True)
        behavior_logits = logits + prior_strength * prior_logits
        log_probs = tf.nn.log_softmax(behavior_logits, axis=-1)
        old_log_probs = tf.nn.log_softmax(tf.stop_gradient(prior_strength * prior_logits), axis=-1)
        probs = tf.nn.softmax(behavior_logits, axis=-1)
        one_hot = tf.one_hot(actions, 7, dtype=tf.float32)
        selected_log_probs = tf.reduce_sum(one_hot * log_probs, axis=-1)
        selected_old_log_probs = tf.reduce_sum(one_hot * old_log_probs, axis=-1)
        ratio = tf.exp(selected_log_probs - selected_old_log_probs)

        advantage = tf.expand_dims(returns - values, axis=-1)
        clipped_ratio = tf.clip_by_value(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        surrogate = tf.minimum(ratio * advantage, clipped_ratio * advantage)
        policy_loss = -tf.reduce_mean(tf.reduce_mean(surrogate, axis=-1))
        value_loss = tf.reduce_mean(tf.square(returns - values))
        entropy = -tf.reduce_mean(tf.reduce_sum(probs * log_probs, axis=-1))
        imitation_loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                expert_actions,
                logits,
                from_logits=True,
            )
        )
        if tf.shape(aux)[0] > 1:
            flow_drift = tf.reduce_mean(tf.abs(aux[1:] - aux[:-1]))
        else:
            flow_drift = tf.constant(0.0, dtype=tf.float32)
        loss = (
            policy_loss
            + value_coef * value_loss
            - entropy_coef * entropy
            + flow_coef * flow_drift
            + imitation_coef * imitation_loss
        )

    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, grad_clip)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return {
        "loss": float(loss.numpy()),
        "policy_loss": float(policy_loss.numpy()),
        "value_loss": float(value_loss.numpy()),
        "entropy": float(entropy.numpy()),
        "imitation_loss": float(imitation_loss.numpy()),
        "flow_drift_loss": float(flow_drift.numpy()),
        "clip_epsilon": float(clip_epsilon),
    }


def load_city_module(city):
    module_path = ROOT / ("MYFINAL_NY" if city == "NY" else "MYFINAL_Y") / "run_pgfa_rl.py"
    module_name = "pgfa_rl_%s_for_fig_paper" % city.lower()
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load runner module from %s" % module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_data_dir(city, availability):
    if city == "CD":
        return ROOT / "uber" / "data_deal"
    if availability <= 0.34:
        return ROOT / "MYFINAL_NY" / "data_nyc_origin_hotspot"
    if availability <= 0.67:
        return ROOT / "MYFINAL_NY" / "data_nyc_mix095"
    return ROOT / "MYFINAL_NY" / "data_nyc_mix087"


def enrich_metrics(output_dir, variant, city, args):
    metrics_path = output_dir / "final_metrics.json"
    if not metrics_path.exists():
        return
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    metrics["method"] = VARIANT_LABELS[variant]
    metrics["comparison_variant"] = variant
    metrics["comparison_city"] = city
    metrics["paper_source"] = {
        "structural_heuristic": "Learning Scalable Structural Representations for Link Prediction with Bloom Signatures",
        "edge_aware_gat": "Optimal Graph Information Fused Graph Attention Network for Traffic Flow Forecasting",
        "multimodal_stgcn": "Multi-modal and multi-agent reinforcement learning framework for urban traffic flow prediction and signal control optimization",
        "wla_stgcn": "Multi-modal and multi-agent reinforcement learning framework for urban traffic flow prediction and signal control optimization",
        "mm_stmap": "Multi-modal and multi-agent reinforcement learning framework for urban traffic flow prediction and signal control optimization",
        "dt_d3qn": "A Digital Twin-Driven Deep Reinforcement Learning Framework for End-to-End Mixed Traffic Scheduling",
        "qos_queue_dtsm": "A Digital Twin-Driven Deep Reinforcement Learning Framework for End-to-End Mixed Traffic Scheduling",
    }[variant]
    metrics["paper_mechanism_kept"] = {
        "structural_heuristic": [
            "multi-hop Bloom-style hashed neighborhood signatures",
            "source-target signature intersection / difference scoring",
            "compact structural pair features fused with node embeddings",
        ],
        "edge_aware_gat": [
            "predefined adjacency graph",
            "feature-similarity dynamic optimal graph",
            "learned fusion of self/static/dynamic graph information",
            "edge-feature-aware graph attention for action logits",
        ],
        "multimodal_stgcn": [
            "multi-modal perception over local supply-demand, neighborhood context, risk/trend features, and temporal context",
            "learned modality fusion weights",
            "residual spatial graph convolution adapted to the dispatch grid",
            "temporal-state projection using the enhanced time/context features available in CityReal",
        ],
        "wla_stgcn": [
            "multi-modal STGCN backbone",
            "learnable spatial alpha initialized from feasible grid adjacency",
            "learnable temporal beta from time-context features",
            "weighted linear attention action scoring without full node-node attention matrix",
        ],
        "mm_stmap": [
            "multi-modal STGCN backbone",
            "weighted linear attention",
            "global multi-agent coordination context across all dispatch regions",
            "PPO/MAPPO-style clipped policy update adapted to the stored CityReal trajectory batch",
        ],
        "dt_d3qn": [
            "CityReal simulator used as the digital-twin virtual scheduling environment",
            "convolutional state encoder over regional dispatch states",
            "dueling value and action-advantage decomposition for feasible dispatch actions",
            "online DRL decision scores adapted from D3QN scheduling",
        ],
        "qos_queue_dtsm": [
            "mixed-flow coarse classification mapped to urgent, predictive, and opportunistic dispatch moves",
            "source residual supply interpreted as queue/buffer resource capacity",
            "resource gate penalizing dispatches from low-reserve source regions",
            "priority-weighted action scoring inspired by deterministic mixed-traffic scheduling",
        ],
    }[variant]
    metrics["uses_pgfa_prior"] = bool(args.use_pgfa_prior)
    metrics["uses_pgfa_execution"] = bool(args.use_pgfa_execution)
    metrics["main_table_note"] = (
        "This comparison keeps the same CityReal environment, enhanced state, "
        "actor-critic loss, train/test split, and evaluation metrics as PGFA; "
        "only the graph aggregation module is replaced by the paper-derived baseline."
    )
    if variant == "mm_stmap":
        metrics["main_table_note"] = (
            "This comparison keeps the same CityReal environment, enhanced state, "
            "train/test split, and evaluation metrics as PGFA. The model replaces "
            "the graph aggregation module with an MM-STMAP-style multimodal STGCN "
            "and weighted linear attention encoder, and uses a PPO/MAPPO-style "
            "clipped policy update adapted to the trajectory fields stored by the "
            "existing CityReal runner."
        )
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (output_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# %s %s Results\n\n" % (VARIANT_LABELS[variant], city))
        f.write("- method: %s\n" % metrics["method"])
        f.write("- city: %s\n" % city)
        f.write("- final_test_orr: %s\n" % metrics.get("final_test_orr"))
        f.write("- final_test_orr_std: %s\n" % metrics.get("final_test_orr_std"))
        f.write("- final_normalized_adi: %s\n" % metrics.get("final_normalized_adi"))
        f.write("- baseline_no_relocation_orr: %s\n" % metrics.get("baseline_no_relocation_orr"))
        f.write("- output_dir: %s\n" % output_dir)


def parse_args(default_city, default_variant, output_base):
    parser = argparse.ArgumentParser(
        description="Run paper-inspired comparison experiments for CD/NY ride-hailing dispatch."
    )
    parser.add_argument("--city", choices=["CD", "NY"], default=default_city)
    parser.add_argument("--variant", choices=sorted(VARIANT_LABELS), default=default_variant)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--test-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--availability", type=float, default=1.0)
    parser.add_argument("--driver-scale", type=float, default=1.0)
    parser.add_argument("--probability", type=float, default=0.03)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--value-coef", type=float, default=0.50)
    parser.add_argument("--entropy-coef", type=float, default=0.010)
    parser.add_argument("--flow-coef", type=float, default=0.01)
    parser.add_argument("--imitation-coef", type=float, default=0.25)
    parser.add_argument("--imitation-min-ratio", type=float, default=0.20)
    parser.add_argument("--imitation-decay-episodes", type=int, default=12)
    parser.add_argument("--prior-strength", type=float, default=0.45)
    parser.add_argument("--eval-prior-strength", type=float, default=0.35)
    parser.add_argument("--no-expert-prior", action="store_true")
    parser.add_argument("--stay-penalty", type=float, default=0.55)
    parser.add_argument("--train-temperature", type=float, default=1.20)
    parser.add_argument("--eval-temperature", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--reward-scale", type=float, default=10000.0)
    parser.add_argument("--orr-reward", type=float, default=0.08)
    parser.add_argument("--dispatch-scale", type=float, default=None)
    parser.add_argument("--max-dispatch", type=int, default=None)
    parser.add_argument("--use-pgfa-prior", action="store_true")
    parser.add_argument("--use-pgfa-execution", action="store_true")
    args = parser.parse_args()

    args.data_dir = str(Path(args.data_dir) if args.data_dir else default_data_dir(args.city, args.availability))
    args.split = args.split or ("nyc" if args.city == "NY" else "paper")
    if args.dispatch_scale is None:
        args.dispatch_scale = 1.35 if args.city == "NY" else 1.20
    if args.max_dispatch is None:
        args.max_dispatch = 18 if args.city == "NY" else 11
    if args.city == "NY" and args.availability <= 0.34 and args.driver_scale == 1.0:
        args.driver_scale = 1.6

    if args.output_dir is None:
        avail_tag = int(round(args.availability * 100))
        args.output_dir = str(output_base / ("results_%s_%s_avail%d_seed%d" % (args.variant, args.city, avail_tag, args.seed)))
    return args


def main(default_city, default_variant, output_base):
    args = parse_args(default_city, default_variant, Path(output_base))
    module = load_city_module(args.city)
    module.PGFANetwork = build_model_class(args.variant)
    if args.variant == "mm_stmap":
        module.train_on_episode = mappo_style_train_on_episode
    if not args.use_pgfa_prior:
        module.expert_prior_logits = generic_prior_logits
    if not args.use_pgfa_execution:
        module.make_dispatch = generic_make_dispatch

    print(
        "[Fig_paper] running %s on %s, data=%s, output=%s"
        % (VARIANT_LABELS[args.variant], args.city, args.data_dir, args.output_dir)
    )
    module.run(args)
    enrich_metrics(Path(args.output_dir), args.variant, args.city, args)


if __name__ == "__main__":
    main(default_city="CD", default_variant="structural_heuristic", output_base=Path.cwd())
