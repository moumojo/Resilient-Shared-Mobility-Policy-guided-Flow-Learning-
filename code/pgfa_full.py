import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import tensorflow as tf
except ImportError as exc:
    tf = None
    _TF_IMPORT_ERROR = exc
else:
    _TF_IMPORT_ERROR = None


EPS = 1.0e-6


CITY_CONFIGS = {
    "CD": {"feature_dim": 17, "hidden_dim": 128, "default_nodes": 232},
    "NY": {"feature_dim": 18, "hidden_dim": 128, "default_nodes": 255},
}


@dataclass
class PGFAConfig:
    rho: float = 0.52
    max_dispatch: int = 7
    top_k: int = 2
    reserve_base: float = 1.0
    reserve_future: float = 0.35
    reserve_neighbor: float = 0.18
    psi_shortage: float = 1.20
    psi_future: float = 0.55
    psi_neighbor: float = 0.35
    psi_supply: float = 0.48
    psi_activity: float = 0.08
    flow_temperature: float = 0.85
    flow_threshold: float = -0.08
    attention_future: float = 0.35
    attention_message: float = 0.30
    source_guard: float = 0.72
    min_source_surplus: float = 1.0
    demand_floor: float = 0.35
    policy_smooth: float = 0.15

    def clipped(self) -> "PGFAConfig":
        values = asdict(self)
        bounds = {
            "rho": (0.10, 0.90),
            "max_dispatch": (1, 15),
            "top_k": (1, 4),
            "reserve_base": (0.0, 5.0),
            "reserve_future": (0.0, 1.5),
            "reserve_neighbor": (0.0, 1.5),
            "psi_shortage": (0.10, 3.0),
            "psi_future": (0.0, 2.0),
            "psi_neighbor": (0.0, 2.0),
            "psi_supply": (0.0, 2.0),
            "psi_activity": (0.0, 1.0),
            "flow_temperature": (0.20, 3.0),
            "flow_threshold": (-1.0, 0.8),
            "attention_future": (0.0, 2.0),
            "attention_message": (0.0, 2.0),
            "source_guard": (0.0, 2.0),
            "min_source_surplus": (0.0, 8.0),
            "demand_floor": (0.0, 4.0),
            "policy_smooth": (0.0, 0.8),
        }
        out = {}
        for key, value in values.items():
            lo, hi = bounds[key]
            if isinstance(getattr(self, key), int):
                out[key] = int(round(float(np.clip(value, lo, hi))))
            else:
                out[key] = float(np.clip(value, lo, hi))
        return PGFAConfig(**out)


@dataclass
class PGFANNConfig:
    hidden_dim: int = 128
    layers: int = 2
    gamma: float = 0.90
    learning_rate: float = 5.0e-4
    value_coef: float = 0.50
    entropy_coef: float = 0.015
    kl_coef: float = 0.020
    flow_coef: float = 0.020
    prior_coef: float = 0.040
    prior_logit_coef: float = 0.75
    rho: float = 0.52
    max_dispatch: int = 7
    top_k: int = 2
    reserve_base: float = 1.0
    reserve_future: float = 0.35
    reserve_neighbor: float = 0.18
    source_guard: float = 0.72
    flow_threshold: float = -0.08
    demand_floor: float = 0.35
    execution_mode: str = "policy_need"
    exec_need_coef: float = 0.55
    exec_flow_coef: float = 1.10
    exec_time_coef: float = 0.20
    execution_aware_loss: bool = True
    greedy_eval: bool = True


def require_tf():
    if tf is None:
        raise ImportError(
            "TensorFlow is required for the neural PGFA implementation. "
            "Please run this script in your TensorFlow environment."
        ) from _TF_IMPORT_ERROR


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def softmax(x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64).copy()
    if mask is not None:
        y = np.where(mask > 0, y, -1.0e9)
    y -= np.max(y)
    exp_y = np.exp(y)
    if mask is not None:
        exp_y *= mask > 0
    denom = np.sum(exp_y)
    if denom <= EPS:
        if mask is None:
            return np.ones_like(y, dtype=np.float64) / max(len(y), 1)
        valid = (mask > 0).astype(np.float64)
        return valid / max(np.sum(valid), 1.0)
    return exp_y / denom


def safe_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - np.mean(x)) / (np.std(x) + 1.0)


def normalize_positive(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    scale = np.percentile(np.abs(x), 90) + 1.0
    return x / scale


def _normalize_count(x: np.ndarray, scale: float) -> np.ndarray:
    return np.asarray(x, dtype=np.float32) / np.float32(max(scale, 1.0))


def _normalize_positive(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    scale = np.percentile(np.abs(x), 90) + 1.0
    return x / np.float32(scale)


def normalize_city(city: str) -> str:
    key = city.upper()
    if key not in CITY_CONFIGS:
        raise ValueError(f"Unknown city {city!r}; expected one of {sorted(CITY_CONFIGS)}")
    return key


def normalize_availability(value: str) -> str:
    aliases = {
        "100": "avail_100",
        "1": "avail_100",
        "1.0": "avail_100",
        "66": "avail_66",
        "0.66": "avail_66",
        "33": "avail_33",
        "0.33": "avail_33",
    }
    key = str(value).strip().lower().replace("%", "")
    if key.startswith("avail_"):
        key = key.replace("avail_", "")
    if key not in aliases:
        raise ValueError("availability must be one of 100, 66, 33, 1.0, 0.66, or 0.33")
    return aliases[key]


def validate_graph(graph: Dict[str, np.ndarray]) -> None:
    action_targets = np.asarray(graph["action_targets"])
    valid_mask = np.asarray(graph["valid_mask"])
    adjacency = np.asarray(graph["adjacency"])
    if action_targets.ndim != 2 or action_targets.shape[1] != 7:
        raise ValueError("action_targets must have shape [N, 7]")
    if valid_mask.shape != action_targets.shape:
        raise ValueError("valid_mask must have the same shape as action_targets")
    n_nodes = action_targets.shape[0]
    if adjacency.shape != (n_nodes, n_nodes):
        raise ValueError("adjacency must have shape [N, N]")


def make_demo_graph(n_nodes: int = 232) -> Dict[str, np.ndarray]:
    action_targets = np.zeros((n_nodes, 7), dtype=np.int32)
    valid_mask = np.ones((n_nodes, 7), dtype=np.float32)
    offsets = np.asarray([1, -1, 2, -2, 3, -3, 0], dtype=np.int32)
    for i in range(n_nodes):
        action_targets[i] = (i + offsets) % n_nodes

    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i in range(n_nodes):
        adjacency[i, i] = 1.0
        for j in action_targets[i, :6]:
            adjacency[i, j] = 1.0
    degree = adjacency.sum(axis=1, keepdims=True)
    adjacency = adjacency / np.maximum(degree, 1.0)
    return {
        "target_grids": np.arange(n_nodes, dtype=np.int32),
        "action_targets": action_targets,
        "valid_mask": valid_mask,
        "adjacency": adjacency,
    }


def load_graph_npz(path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    graph = {
        "action_targets": np.asarray(data["action_targets"], dtype=np.int32),
        "valid_mask": np.asarray(data["valid_mask"], dtype=np.float32),
        "adjacency": np.asarray(data["adjacency"], dtype=np.float32),
    }
    if "target_grids" in data:
        graph["target_grids"] = np.asarray(data["target_grids"], dtype=np.int32)
    else:
        graph["target_grids"] = np.arange(graph["action_targets"].shape[0], dtype=np.int32)
    if "edge_attr" in data:
        graph["edge_attr"] = np.asarray(data["edge_attr"], dtype=np.float32)
    validate_graph(graph)
    return graph


def build_graph(env) -> Dict[str, np.ndarray]:
    target_grids = np.asarray(env.target_grids, dtype=np.int32)
    grid_to_idx = {int(grid_id): idx for idx, grid_id in enumerate(target_grids)}
    n_nodes = len(target_grids)
    action_targets = np.zeros((n_nodes, 7), dtype=np.int32)
    valid_mask = np.zeros((n_nodes, 7), dtype=np.float64)
    edge_distance = np.zeros((n_nodes, 7), dtype=np.float64)
    edge_travel_time = np.zeros((n_nodes, 7), dtype=np.float64)
    neighbors: List[List[int]] = []

    for i, node_id in enumerate(target_grids):
        action_targets[i, :] = i
        valid_mask[i, 6] = 1.0
        src_row = int(node_id) // int(env.N)
        src_col = int(node_id) % int(env.N)
        local = [i]
        node = env.nodes[int(node_id)]
        for action_id, neighbor in enumerate(node.neighbors[:6]):
            if neighbor is None:
                continue
            neighbor_id = int(neighbor.get_node_index())
            j = grid_to_idx.get(neighbor_id)
            if j is None:
                continue
            action_targets[i, action_id] = j
            valid_mask[i, action_id] = 1.0
            dst_row = neighbor_id // int(env.N)
            dst_col = neighbor_id % int(env.N)
            grid_distance = math.sqrt(float((dst_row - src_row) ** 2 + (dst_col - src_col) ** 2))
            edge_distance[i, action_id] = grid_distance
            edge_travel_time[i, action_id] = 10.0 * max(grid_distance, 1.0)
            local.append(j)
        neighbors.append(sorted(set(local)))

    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i, local in enumerate(neighbors):
        adjacency[i, local] = 1.0
    degree = np.sum(adjacency, axis=1, keepdims=True)
    degree[degree <= 0] = 1.0
    adjacency = adjacency / degree
    max_distance = max(float(np.max(edge_distance)), 1.0)
    max_travel_time = max(float(np.max(edge_travel_time)), 10.0)
    edge_attr = np.stack(
        [
            valid_mask,
            (action_targets == np.arange(n_nodes)[:, None]).astype(np.float64),
            edge_distance / max_distance,
            edge_travel_time / max_travel_time,
        ],
        axis=-1,
    )

    return {
        "target_grids": target_grids,
        "action_targets": action_targets,
        "valid_mask": valid_mask,
        "edge_attr": edge_attr,
        "edge_distance": edge_distance,
        "edge_travel_time": edge_travel_time,
        "neighbors": neighbors,
        "adjacency": adjacency,
    }


def flatten_state(env, state) -> Tuple[np.ndarray, np.ndarray]:
    target = np.asarray(env.target_grids, dtype=np.int32)
    drivers = np.asarray(state[0], dtype=np.float64).reshape(-1)[target]
    orders = np.asarray(state[1], dtype=np.float64).reshape(-1)[target]
    return drivers, orders


def demand_from_order_num(order_num_dist, env, graph, offset: int = 1) -> np.ndarray:
    moment = int((env.city_time + offset) % 144)
    row = order_num_dist[moment]
    demand = np.zeros(len(graph["target_grids"]), dtype=np.float64)
    if not isinstance(row, dict):
        return demand

    grid_to_idx = {int(grid_id): idx for idx, grid_id in enumerate(graph["target_grids"])}
    for raw_id, value in row.items():
        idx = grid_to_idx.get(int(raw_id))
        if idx is None:
            continue
        if isinstance(value, (list, tuple, np.ndarray)):
            demand[idx] = float(value[0]) if len(value) > 0 else 0.0
        else:
            demand[idx] = float(value)
    return demand


def build_enhanced_state(env, state, order_num_dist, graph) -> Dict[str, np.ndarray]:
    drivers, orders = flatten_state(env, state)
    drivers = drivers.astype(np.float32)
    orders = orders.astype(np.float32)
    adjacency = np.asarray(graph["adjacency"], dtype=np.float32)

    next_1 = demand_from_order_num(order_num_dist, env, graph, offset=1).astype(np.float32)
    next_2 = demand_from_order_num(order_num_dist, env, graph, offset=2).astype(np.float32)
    future = 0.70 * next_1 + 0.30 * next_2

    neigh_driver = adjacency.dot(drivers)
    neigh_order = adjacency.dot(orders)
    shortage = np.maximum(orders - drivers, 0.0)
    surplus = np.maximum(drivers - orders, 0.0)
    neigh_shortage = adjacency.dot(shortage)

    volume_scale = float(np.percentile(drivers + orders + future, 90) + 1.0)
    d_scaled = _normalize_count(drivers, volume_scale)
    o_scaled = _normalize_count(orders, volume_scale)
    nd_scaled = _normalize_count(neigh_driver, volume_scale)
    no_scaled = _normalize_count(neigh_order, volume_scale)
    activity = 0.5 * (drivers + orders)

    rsd = drivers / (orders + 1.0)
    balance = (drivers - orders) / (drivers + orders + 1.0)
    gap = (orders - drivers) / (orders + 1.0)
    urgency = np.maximum(gap, 0.0)
    d_rel = drivers / (np.mean(drivers) + 1.0)
    o_rel = orders / (np.mean(orders) + 1.0)
    r_neigh = neigh_driver / (neigh_order + 1.0)
    potential = np.minimum(drivers, orders) / (orders + 1.0)
    priority = orders / (rsd + 1.0)

    if len(urgency) > 0:
        p75 = np.percentile(urgency, 75)
        p25 = np.percentile(urgency, 25)
        trend = np.where(urgency > p75, 1.0, np.where(urgency < p25, -1.0, 0.0))
    else:
        trend = np.zeros_like(urgency)

    t = int(env.city_time % 144)
    t_norm = np.full_like(drivers, t / 143.0, dtype=np.float32)
    t_sin = np.full_like(drivers, np.sin(2.0 * np.pi * t / 144.0), dtype=np.float32)
    t_cos = np.full_like(drivers, np.cos(2.0 * np.pi * t / 144.0), dtype=np.float32)
    weekend = np.zeros_like(drivers, dtype=np.float32)
    peak = np.full_like(drivers, 1.0 if (36 <= t <= 54 or 96 <= t <= 114) else 0.0, dtype=np.float32)

    features = np.stack(
        [
            d_scaled,
            o_scaled,
            np.clip(rsd / 8.0, 0.0, 4.0),
            np.clip(balance, -1.0, 1.0),
            np.clip(urgency, 0.0, 4.0),
            np.clip(d_rel / 4.0, 0.0, 4.0),
            np.clip(o_rel / 4.0, 0.0, 4.0),
            nd_scaled,
            no_scaled,
            np.clip(r_neigh / 8.0, 0.0, 4.0),
            trend.astype(np.float32),
            np.clip(potential, 0.0, 1.0),
            np.clip(gap, -4.0, 4.0),
            _normalize_count(priority, volume_scale),
            _normalize_count(activity, volume_scale),
            t_norm,
            t_sin,
            t_cos,
            weekend,
            peak,
        ],
        axis=1,
    ).astype(np.float32)

    reserve = (
        np.float32(1.0)
        + np.float32(0.35) * _normalize_positive(future)
        + np.float32(0.18) * _normalize_positive(neigh_shortage)
    )
    pressure = np.stack(
        [
            _normalize_positive(shortage),
            _normalize_positive(future),
            _normalize_positive(neigh_shortage),
            _normalize_positive(surplus),
            reserve.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    return {
        "x": features,
        "pressure": pressure,
        "drivers": drivers,
        "orders": orders,
        "future": future,
        "shortage": shortage,
        "surplus": surplus,
        "neigh_shortage": neigh_shortage,
        "volume_scale": np.float32(volume_scale),
    }


class PGFAController:
    def __init__(self, graph: Dict[str, np.ndarray], config: Optional[PGFAConfig] = None):
        self.graph = graph
        self.config = config or PGFAConfig()
        self._ema_psi: Optional[np.ndarray] = None
        self._prev_psi: Optional[np.ndarray] = None

    def reset_episode(self):
        self._ema_psi = None
        self._prev_psi = None

    def features(self, env, state, order_num_dist) -> Dict[str, np.ndarray]:
        drivers, orders = flatten_state(env, state)
        next_1 = demand_from_order_num(order_num_dist, env, self.graph, offset=1)
        next_2 = demand_from_order_num(order_num_dist, env, self.graph, offset=2)
        future = 0.70 * next_1 + 0.30 * next_2
        adjacency = self.graph["adjacency"]
        neigh_driver = adjacency.dot(drivers)
        neigh_order = adjacency.dot(orders)
        shortage = np.maximum(orders - drivers, 0.0)
        surplus = np.maximum(drivers - orders, 0.0)
        neigh_shortage = adjacency.dot(shortage)
        activity = drivers + orders
        return {
            "drivers": drivers,
            "orders": orders,
            "future": future,
            "neigh_driver": neigh_driver,
            "neigh_order": neigh_order,
            "shortage": shortage,
            "surplus": surplus,
            "neigh_shortage": neigh_shortage,
            "activity": activity,
        }

    def policy_flow_prior(self, feat: Dict[str, np.ndarray]) -> np.ndarray:
        c = self.config
        shortage = normalize_positive(feat["shortage"])
        future = normalize_positive(feat["future"])
        neigh_shortage = normalize_positive(feat["neigh_shortage"])
        surplus = normalize_positive(feat["surplus"])
        activity = safe_zscore(feat["activity"])
        psi_raw = (
            c.psi_shortage * shortage
            + c.psi_future * future
            + c.psi_neighbor * neigh_shortage
            + c.psi_activity * activity
            - c.psi_supply * surplus
        )
        psi = np.tanh(psi_raw)
        if self._ema_psi is None:
            self._ema_psi = psi
        else:
            alpha = c.policy_smooth
            self._ema_psi = alpha * self._ema_psi + (1.0 - alpha) * psi
        return self._ema_psi.copy()

    def policy_guided_message(self, feat: Dict[str, np.ndarray], psi: np.ndarray) -> Dict[str, np.ndarray]:
        adjacency = self.graph["adjacency"]
        flow_in = adjacency.dot(np.maximum(psi, 0.0))
        flow_out = adjacency.dot(np.maximum(-psi, 0.0))
        demand_message = normalize_positive(feat["neigh_order"] + feat["future"] + feat["neigh_shortage"])
        supply_message = normalize_positive(feat["neigh_driver"] + feat["surplus"])
        pg_message = np.tanh(demand_message - 0.55 * supply_message + flow_in - 0.35 * flow_out)
        return {
            "flow_in": flow_in,
            "flow_out": flow_out,
            "pg_message": pg_message,
        }

    def flow_aware_attention(
        self,
        source: int,
        feat: Dict[str, np.ndarray],
        psi: np.ndarray,
        message: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        c = self.config
        logits = np.full(7, -1.0e9, dtype=np.float64)
        mask = self.graph["valid_mask"][source]
        action_targets = self.graph["action_targets"][source]
        source_surplus = feat["surplus"][source]
        source_cost = (
            feat["shortage"][source]
            + c.reserve_future * feat["future"][source]
            + c.reserve_neighbor * feat["neigh_shortage"][source]
        )
        scale = np.mean(feat["drivers"] + feat["orders"]) + 1.0
        source_gate = sigmoid((source_surplus - c.source_guard * source_cost) / scale)

        for action_id in range(6):
            if mask[action_id] <= 0:
                continue
            target = int(action_targets[action_id])
            if target == source:
                continue
            f_ij = psi[target] - psi[source]
            target_need = (
                feat["shortage"][target]
                + c.attention_future * feat["future"][target]
                + c.attention_message * np.maximum(message["pg_message"][target], 0.0)
            )
            if target_need <= c.demand_floor and f_ij <= 0:
                continue
            direction_gate = sigmoid((f_ij - c.flow_threshold) / max(c.flow_temperature, EPS))
            logits[action_id] = (
                math.log(max(target_need, EPS))
                + 1.35 * f_ij
                + math.log(max(direction_gate, EPS))
                + math.log(max(source_gate, EPS))
            )
        logits[6] = 0.0
        prob = softmax(logits, mask)
        return logits, prob

    def dispatch(self, env, state, order_num_dist):
        feat = self.features(env, state, order_num_dist)
        psi = self.policy_flow_prior(feat)
        message = self.policy_guided_message(feat, psi)
        actions = []
        action_prob_sum = np.zeros(7, dtype=np.float64)
        n_policy_rows = 0
        f_values = []

        for source in range(len(self.graph["target_grids"])):
            source_surplus = feat["surplus"][source]
            reserve = (
                self.config.reserve_base
                + self.config.reserve_future * feat["future"][source]
                + self.config.reserve_neighbor * feat["neigh_shortage"][source]
            )
            available = int(math.floor(source_surplus - reserve))
            if available <= self.config.min_source_surplus:
                continue

            logits, prob = self.flow_aware_attention(source, feat, psi, message)
            action_prob_sum += prob
            n_policy_rows += 1
            candidate_ids = [a for a in range(6) if logits[a] > -1.0e8]
            candidate_ids = sorted(candidate_ids, key=lambda a: logits[a], reverse=True)
            remaining = min(available, self.config.max_dispatch)

            for action_id in candidate_ids[: self.config.top_k]:
                if remaining <= 0:
                    break
                target = int(self.graph["action_targets"][source, action_id])
                f_ij = psi[target] - psi[source]
                target_need = feat["shortage"][target] + self.config.attention_future * feat["future"][target]
                if target_need <= self.config.demand_floor and f_ij <= 0:
                    continue
                intensity = sigmoid(logits[action_id])
                num = int(math.floor(self.config.rho * min(remaining, max(target_need, 1.0)) * intensity))
                num = max(0, min(int(num), remaining))
                if num <= 0:
                    continue
                actions.append(
                    (
                        int(self.graph["target_grids"][source]),
                        int(self.graph["target_grids"][target]),
                        int(num),
                    )
                )
                f_values.append(float(f_ij))
                remaining -= num

        if n_policy_rows > 0:
            mean_policy = action_prob_sum / n_policy_rows
        else:
            mean_policy = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64)

        if self._prev_psi is None:
            flow_drift = 0.0
        else:
            flow_drift = float(np.mean(np.abs(psi - self._prev_psi)))
        self._prev_psi = psi.copy()
        diagnostics = {
            "mean_policy": mean_policy,
            "flow_mean": float(np.mean(f_values)) if f_values else 0.0,
            "flow_abs": float(np.mean(np.abs(f_values))) if f_values else 0.0,
            "flow_drift": flow_drift,
            "num_actions": len(actions),
            "psi_mean": float(np.mean(psi)),
            "psi_std": float(np.std(psi)),
        }
        return actions, diagnostics


class PGFATuner:
    float_keys = [
        "rho",
        "reserve_base",
        "reserve_future",
        "reserve_neighbor",
        "psi_shortage",
        "psi_future",
        "psi_neighbor",
        "psi_supply",
        "psi_activity",
        "flow_temperature",
        "flow_threshold",
        "attention_future",
        "attention_message",
        "source_guard",
        "min_source_surplus",
        "demand_floor",
        "policy_smooth",
    ]
    int_keys = ["max_dispatch", "top_k"]

    def __init__(self, base_config: PGFAConfig, seed: int = 2022, noise: float = 0.10):
        self.rng = np.random.RandomState(seed)
        self.best_config = base_config.clipped()
        self.current_config = self.best_config
        self.best_score = -1.0e18
        self.noise = float(noise)

    def score(self, episode_result: Dict[str, float]) -> float:
        return (
            100.0 * episode_result["orr"]
            + 0.015 * episode_result["adi"]
            - 0.02 * episode_result.get("flow_drift", 0.0)
        )

    def update(self, episode_result: Dict[str, float]) -> PGFAConfig:
        score = self.score(episode_result)
        if score >= self.best_score:
            self.best_score = score
            self.best_config = self.current_config
        center = self.best_config
        values = asdict(center)
        scale = self.noise * (0.75 + 0.25 * self.rng.rand())
        for key in self.float_keys:
            base = float(values[key])
            jitter = self.rng.normal(0.0, scale * max(abs(base), 0.20))
            values[key] = base + jitter
        for key in self.int_keys:
            values[key] = int(round(values[key] + self.rng.choice([-1, 0, 1])))
        self.current_config = PGFAConfig(**values).clipped()
        return self.current_config


def categorical_kl(p: Sequence[float], q: Sequence[float]) -> float:
    p = np.asarray(p, dtype=np.float64) + EPS
    q = np.asarray(q, dtype=np.float64) + EPS
    p /= np.sum(p)
    q /= np.sum(q)
    return float(np.sum(p * np.log(p / q)))


class StateEncoder(tf.keras.layers.Layer if tf is not None else object):
    def __init__(self, hidden_dim: int):
        require_tf()
        super().__init__()
        self.blocks = [
            tf.keras.layers.Dense(hidden_dim, activation="relu", name="dense5_%d" % i)
            for i in range(5)
        ]

    def call(self, x, training=False):
        h = tf.convert_to_tensor(x, dtype=tf.float32)
        for block in self.blocks:
            h = block(h, training=training)
        return h


class PolicyFlowPrior(tf.keras.layers.Layer if tf is not None else object):
    def __init__(self, hidden_dim: int):
        require_tf()
        super().__init__(name="policy_flow_prior")
        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(hidden_dim // 2, activation="relu"),
                tf.keras.layers.Dense(1, activation="tanh"),
            ],
            name="policy_flow_prior",
        )

    def call(self, h, pressure, training=False):
        psi_in = tf.concat([h, pressure], axis=-1)
        return tf.squeeze(self.net(psi_in, training=training), axis=-1)


class PolicyGuidedMessage(tf.keras.layers.Layer if tf is not None else object):
    def __init__(self, hidden_dim: int):
        require_tf()
        super().__init__(name="policy_guided_message")
        self.message_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(hidden_dim, activation="relu"),
                tf.keras.layers.Dense(hidden_dim, activation="relu"),
            ],
            name="policy_guided_message",
        )

    def call(self, target_h, f_ij, edge_attr, training=False):
        message_in = tf.concat([target_h, tf.expand_dims(f_ij, axis=-1), edge_attr], axis=-1)
        return self.message_mlp(message_in, training=training)


class FlowAwareAttention(tf.keras.layers.Layer if tf is not None else object):
    def __init__(self, hidden_dim: int):
        require_tf()
        super().__init__(name="flow_aware_attention")
        self.attention_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(hidden_dim, activation="relu"),
                tf.keras.layers.Dense(1),
            ],
            name="flow_aware_attention",
        )

    def call(self, source_h, target_h, messages, f_ij, valid_mask, training=False):
        attention_in = tf.concat([source_h, target_h, messages, tf.expand_dims(f_ij, axis=-1)], axis=-1)
        attention_logits = tf.squeeze(self.attention_mlp(attention_in, training=training), axis=-1)
        return attention_logits + (1.0 - valid_mask) * -1.0e9


class PGFALayer(tf.keras.layers.Layer if tf is not None else object):
    def __init__(self, hidden_dim: int, action_targets, valid_mask, edge_attr, name: str):
        require_tf()
        super().__init__(name=name)
        self.action_targets = action_targets
        self.valid_mask = valid_mask
        self.edge_attr = edge_attr
        self.message = PolicyGuidedMessage(hidden_dim)
        self.attention = FlowAwareAttention(hidden_dim)
        self.update = tf.keras.layers.Dense(hidden_dim, activation="relu", name=name + "_update")

    def call(self, h, psi, training=False):
        source_h = tf.tile(tf.expand_dims(h, axis=1), [1, 7, 1])
        target_h = tf.gather(h, self.action_targets)
        target_psi = tf.gather(psi, self.action_targets)
        f_ij = target_psi - tf.expand_dims(psi, axis=1)
        messages = self.message(target_h, f_ij, self.edge_attr, training=training)
        attention_logits = self.attention(source_h, target_h, messages, f_ij, self.valid_mask, training=training)
        attention = tf.nn.softmax(attention_logits, axis=-1)
        aggregated = tf.reduce_sum(tf.expand_dims(attention, axis=-1) * messages, axis=1)
        h_next = self.update(tf.concat([h, aggregated], axis=-1), training=training)
        return h_next, attention_logits, messages


class ActorCriticHeads(tf.keras.layers.Layer if tf is not None else object):
    def __init__(self):
        require_tf()
        super().__init__(name="actor_critic_heads")
        self.policy_head = tf.keras.layers.Dense(7, name="policy_head")
        self.value_head = tf.keras.layers.Dense(1, name="value_head")

    def call(self, h, attention_logits, valid_mask, training=False):
        policy_logits = self.policy_head(h, training=training)
        if attention_logits is not None:
            policy_logits = policy_logits + 0.50 * attention_logits
        policy_logits = policy_logits + (1.0 - valid_mask) * -1.0e9
        values = tf.squeeze(self.value_head(h, training=training), axis=-1)
        return policy_logits, values


class PGFANetwork(tf.keras.Model if tf is not None else object):
    def __init__(self, graph: Dict[str, np.ndarray], config: PGFANNConfig):
        require_tf()
        super().__init__()
        self.config = config
        self.action_targets = tf.constant(graph["action_targets"], dtype=tf.int32)
        self.valid_mask = tf.constant(graph["valid_mask"], dtype=tf.float32)
        if "edge_attr" in graph:
            self.edge_attr = tf.constant(graph["edge_attr"], dtype=tf.float32)
        else:
            self_mask = tf.cast(
                tf.equal(self.action_targets, tf.expand_dims(tf.range(self.action_targets.shape[0]), 1)),
                tf.float32,
            )
            self.edge_attr = tf.stack([self.valid_mask, self_mask], axis=-1)
        self.encoder = StateEncoder(config.hidden_dim)
        self.policy_flow_prior = PolicyFlowPrior(config.hidden_dim)
        self.update_layers = [
            PGFALayer(
                config.hidden_dim,
                self.action_targets,
                self.valid_mask,
                self.edge_attr,
                name="pgfa_layer_%d" % k,
            )
            for k in range(config.layers)
        ]
        self.heads = ActorCriticHeads()

    def call(self, x, pressure, training=False):
        pressure = tf.convert_to_tensor(pressure, dtype=tf.float32)
        h = self.encoder(x, training=training)
        psi = self.policy_flow_prior(h, pressure, training=training)
        last_attention_logits = None
        last_messages = None

        for layer in self.update_layers:
            h, last_attention_logits, messages = layer(h, psi, training=training)
            last_messages = messages

        policy_logits, values = self.heads(h, last_attention_logits, self.valid_mask, training=training)
        return {
            "logits": policy_logits,
            "values": values,
            "psi": psi,
            "messages": last_messages,
        }


class PGFACheckpointNetwork(tf.keras.Model if tf is not None else object):
    def __init__(
        self,
        graph: Dict[str, np.ndarray],
        hidden_dim: int = 128,
        entropy_floor: float = -1.0e9,
        ablation: str = "full",
    ):
        require_tf()
        super().__init__(name="PGFA_actor_critic")
        validate_graph(graph)
        self.action_targets = tf.constant(graph["action_targets"], dtype=tf.int32)
        self.valid_mask = tf.constant(graph["valid_mask"], dtype=tf.float32)
        self.adjacency = tf.constant(graph["adjacency"], dtype=tf.float32)
        self.entropy_floor = entropy_floor
        self.ablation = ablation
        self.enc1 = tf.keras.layers.Dense(hidden_dim, activation="relu", name="state_encoder_1")
        self.enc2 = tf.keras.layers.Dense(hidden_dim, activation="relu", name="state_encoder_2")
        self.psi_head = tf.keras.layers.Dense(1, activation="tanh", name="policy_flow_prior")
        self.message_dense = tf.keras.layers.Dense(hidden_dim, activation="relu", name="policy_guided_message")
        self.att_dense = tf.keras.layers.Dense(hidden_dim, activation="relu", name="flow_attention_hidden")
        self.logit_head = tf.keras.layers.Dense(1, activation=None, name="action_logit")
        self.value_dense1 = tf.keras.layers.Dense(hidden_dim, activation="relu", name="value_hidden")
        self.value_out = tf.keras.layers.Dense(1, activation=None, name="value")

    def _encode_states(self, node_features):
        h = self.enc1(node_features)
        h = self.enc2(h)
        neigh_h = tf.einsum("ij,bjh->bih", self.adjacency, h)
        return 0.65 * h + 0.35 * neigh_h

    def _policy_flow_prior(self, h):
        psi = self.psi_head(h)
        if self.ablation == "no_policy_flow_prior":
            psi = tf.zeros_like(psi)
        h_j = tf.gather(h, self.action_targets, axis=1)
        psi_j = tf.gather(psi, self.action_targets, axis=1)
        h_i = tf.tile(tf.expand_dims(h, axis=2), [1, 1, 7, 1])
        psi_i = tf.tile(tf.expand_dims(psi, axis=2), [1, 1, 7, 1])
        f_ij = psi_j - psi_i
        return psi, h_i, h_j, f_ij

    def _policy_guided_message(self, h_j, f_ij):
        message_flow = tf.zeros_like(f_ij) if self.ablation == "no_policy_guided_message" else f_ij
        message = self.message_dense(tf.concat([h_j, message_flow], axis=-1))
        if self.ablation == "no_policy_guided_message":
            message = tf.zeros_like(message)
        return message

    def _flow_aware_action_logits(self, h_i, h_j, message, f_ij):
        attention_flow = tf.zeros_like(f_ij) if self.ablation == "no_flow_aware_attention" else f_ij
        attention_input = tf.concat([h_i, h_j, message, attention_flow], axis=-1)
        att_hidden = self.att_dense(attention_input)
        logits = tf.squeeze(self.logit_head(att_hidden), axis=-1)
        return logits + (1.0 - self.valid_mask[None, :, :]) * self.entropy_floor

    def _critic_value(self, h, psi):
        pooled = tf.reduce_mean(tf.concat([h, psi], axis=-1), axis=1)
        return tf.squeeze(self.value_out(self.value_dense1(pooled)), axis=-1)

    def call(self, node_features, training: bool = False):
        h = self._encode_states(node_features)
        psi, h_i, h_j, f_ij = self._policy_flow_prior(h)
        message = self._policy_guided_message(h_j, f_ij)
        logits = self._flow_aware_action_logits(h_i, h_j, message, f_ij)
        value = self._critic_value(h, psi)
        return logits, value, psi


def build_model(
    city: str,
    graph: Optional[Dict[str, np.ndarray]] = None,
    ablation: str = "full",
) -> Tuple[PGFACheckpointNetwork, int]:
    city = normalize_city(city)
    cfg = CITY_CONFIGS[city]
    if graph is None:
        graph = make_demo_graph(cfg["default_nodes"])
    validate_graph(graph)
    model = PGFACheckpointNetwork(graph, hidden_dim=cfg["hidden_dim"], ablation=ablation)
    n_nodes = graph["action_targets"].shape[0]
    feature_dim = cfg["feature_dim"]
    _ = model(tf.zeros((1, n_nodes, feature_dim), dtype=tf.float32), training=False)
    return model, feature_dim


def checkpoint_prefix(weights_root, city: str, availability: str):
    from pathlib import Path

    city = normalize_city(city)
    availability_dir = normalize_availability(availability)
    return Path(weights_root) / city / availability_dir / "best_pgfa_rl_weights"


def load_pretrained(
    city: str,
    availability: str,
    weights_root="weights",
    graph: Optional[Dict[str, np.ndarray]] = None,
    ablation: str = "full",
) -> Tuple[PGFACheckpointNetwork, int, object]:
    model, feature_dim = build_model(city, graph=graph, ablation=ablation)
    prefix = checkpoint_prefix(weights_root, city, availability)
    if not prefix.with_suffix(".index").exists():
        raise FileNotFoundError(f"Missing checkpoint index: {prefix}.index")
    status = model.load_weights(str(prefix))
    try:
        status.assert_existing_objects_matched()
        status.expect_partial()
    except AttributeError:
        pass
    return model, feature_dim, prefix


def policy_from_logits(logits):
    return tf.nn.softmax(logits, axis=-1)


class NeuralPGFAAgent:
    def __init__(self, graph: Dict[str, np.ndarray], config: Optional[PGFANNConfig] = None, seed: int = 2022):
        require_tf()
        self.graph = graph
        self.config = config or PGFANNConfig()
        self.model = PGFANetwork(graph, self.config)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        self.rng = np.random.RandomState(seed)
        self.prev_psi: Optional[np.ndarray] = None
        self.prev_policy_probs: Optional[np.ndarray] = None
        dummy_x = np.zeros((len(graph["target_grids"]), 20), dtype=np.float32)
        dummy_p = np.zeros((len(graph["target_grids"]), 5), dtype=np.float32)
        self.model(dummy_x, dummy_p, training=False)

    def reset_episode(self):
        self.prev_psi = None
        self.prev_policy_probs = None

    def _reserve(self, feat: Dict[str, np.ndarray]) -> np.ndarray:
        c = self.config
        return (
            c.reserve_base
            + c.reserve_future * _normalize_positive(feat["future"])
            + c.reserve_neighbor * _normalize_positive(feat["neigh_shortage"])
        )

    def _execution_score(self, source: int, action_id: int, feat, psi, action_scores) -> float:
        c = self.config
        base_score = float(action_scores[source, action_id])
        if c.execution_mode == "policy":
            return base_score

        target = int(self.graph["action_targets"][source, action_id])
        f_ij = float(psi[target] - psi[source])
        target_need = float(feat["shortage"][target] + c.reserve_future * feat["future"][target])
        need_score = np.log1p(max(target_need, 0.0))

        if "edge_attr" in self.graph and self.graph["edge_attr"].shape[-1] >= 4:
            travel_penalty = float(self.graph["edge_attr"][source, action_id, 3])
        elif "edge_travel_time" in self.graph:
            edge_time = self.graph["edge_travel_time"]
            travel_penalty = float(edge_time[source, action_id] / max(float(np.max(edge_time)), 1.0))
        else:
            travel_penalty = 0.0

        return (
            base_score
            + c.exec_need_coef * need_score
            + c.exec_flow_coef * f_ij
            - c.exec_time_coef * travel_penalty
        )

    def _make_actions(self, feat, action_ids, psi, action_scores, training: bool) -> Tuple[list, Dict[str, float], np.ndarray]:
        c = self.config
        actions = []
        f_values = []
        action_targets = self.graph["action_targets"]
        target_grids = self.graph["target_grids"]
        valid_mask = self.graph["valid_mask"]
        reserve = self._reserve(feat)
        executed_weight = np.zeros_like(action_scores, dtype=np.float32)

        for source, action_id in enumerate(np.asarray(action_ids, dtype=np.int32)):
            source_surplus = float(feat["surplus"][source])
            source_cost = float(
                feat["shortage"][source]
                + c.reserve_future * feat["future"][source]
                + c.reserve_neighbor * feat["neigh_shortage"][source]
            )
            if source_surplus <= c.source_guard * source_cost:
                continue
            available = int(np.floor(source_surplus - reserve[source]))
            if available <= 0:
                continue

            candidate_ids = [a for a in range(6) if valid_mask[source, a] > 0]
            candidate_ids = sorted(
                candidate_ids,
                key=lambda a: self._execution_score(source, a, feat, psi, action_scores),
                reverse=True,
            )
            if training and 0 <= int(action_id) < 6 and int(action_id) in candidate_ids:
                candidate_ids.remove(int(action_id))
                candidate_ids.insert(0, int(action_id))

            remaining = min(int(available), int(c.max_dispatch))
            used_targets = set()
            for candidate_id in candidate_ids[: max(int(c.top_k), 1)]:
                if remaining <= 0:
                    break
                target = int(action_targets[source, candidate_id])
                if target == source or target in used_targets:
                    continue
                f_ij = float(psi[target] - psi[source])
                target_need = float(feat["shortage"][target] + c.reserve_future * feat["future"][target])
                if target_need <= c.demand_floor and f_ij <= c.flow_threshold:
                    continue
                score = float(np.clip(self._execution_score(source, candidate_id, feat, psi, action_scores), -40.0, 40.0))
                intensity = 1.0 / (1.0 + np.exp(-score))
                dispatch_budget = min(float(remaining), max(target_need, 1.0))
                num = int(np.floor(c.rho * dispatch_budget * max(intensity, 0.35)))
                num = max(0, min(num, remaining))
                if num <= 0:
                    continue
                actions.append((int(target_grids[source]), int(target_grids[target]), int(num)))
                f_values.append(f_ij)
                executed_weight[source, candidate_id] += float(num) / max(float(c.max_dispatch), 1.0)
                used_targets.add(target)
                remaining -= num

        if self.prev_psi is None:
            flow_drift = 0.0
        else:
            flow_drift = float(np.mean(np.abs(psi - self.prev_psi)))
        self.prev_psi = np.asarray(psi, dtype=np.float32).copy()
        return actions, {
            "num_actions": float(len(actions)),
            "flow_abs": float(np.mean(np.abs(f_values))) if f_values else 0.0,
            "flow_drift": flow_drift,
            "psi_std": float(np.std(psi)),
        }, executed_weight

    def select_action(self, env, state, order_num_dist, training: bool = True):
        feat = build_enhanced_state(env, state, order_num_dist, self.graph)
        out = self.model(feat["x"], feat["pressure"], training=training)
        logits = out["logits"]
        values = out["values"]
        psi = out["psi"]
        source_active = tf.convert_to_tensor((feat["surplus"] > 0).astype(np.float32))
        pressure = tf.convert_to_tensor(feat["pressure"], dtype=tf.float32)
        prior_node = (
            pressure[:, 0]
            + 0.50 * pressure[:, 1]
            + 0.35 * pressure[:, 2]
            - 0.45 * pressure[:, 3]
            + 0.10 * pressure[:, 4]
        )
        prior_logits = tf.gather(prior_node, self.model.action_targets) - tf.expand_dims(prior_node, axis=1)
        prior_logits = prior_logits + (1.0 - self.model.valid_mask) * -1.0e9
        guided_logits = logits + self.config.prior_logit_coef * prior_logits
        guided_logits = guided_logits + (1.0 - self.model.valid_mask) * -1.0e9
        log_probs = tf.nn.log_softmax(guided_logits, axis=-1)
        probs = tf.nn.softmax(guided_logits, axis=-1)
        prior_probs = tf.stop_gradient(tf.nn.softmax(prior_logits, axis=-1))
        prior_loss_by_node = -tf.reduce_sum(prior_probs * log_probs, axis=-1)

        if self.prev_policy_probs is None:
            kl_by_node = tf.zeros_like(source_active)
        else:
            old_probs = tf.convert_to_tensor(self.prev_policy_probs, dtype=tf.float32)
            old_probs = old_probs / (tf.reduce_sum(old_probs, axis=-1, keepdims=True) + 1.0e-8)
            kl_by_node = tf.reduce_sum(old_probs * (tf.math.log(old_probs + 1.0e-8) - log_probs), axis=-1)

        if training:
            sampled = tf.squeeze(tf.random.categorical(guided_logits, 1), axis=1)
        elif self.config.greedy_eval:
            sampled = tf.argmax(guided_logits, axis=1, output_type=tf.int64)
        else:
            sampled = tf.squeeze(tf.random.categorical(guided_logits, 1), axis=1)

        one_hot = tf.one_hot(sampled, 7, dtype=tf.float32)
        selected_log_prob = tf.reduce_sum(one_hot * log_probs, axis=-1)
        entropy = -tf.reduce_sum(probs * log_probs, axis=-1)
        denom = tf.reduce_sum(source_active) + 1.0e-6
        step_entropy = tf.reduce_sum(entropy * source_active) / denom
        step_prior_loss = tf.reduce_sum(prior_loss_by_node * source_active) / denom
        step_kl_loss = tf.reduce_sum(kl_by_node * source_active) / denom
        step_value = tf.reduce_mean(values)
        action_ids = sampled.numpy()
        psi_np = psi.numpy()
        actions, diag, executed_weight = self._make_actions(feat, action_ids, psi_np, guided_logits.numpy(), training=training)

        if training and self.config.execution_aware_loss and float(np.sum(executed_weight)) > 0.0:
            executed_weight_tf = tf.convert_to_tensor(executed_weight, dtype=tf.float32)
            step_log_prob = tf.reduce_sum(log_probs * executed_weight_tf) / (tf.reduce_sum(executed_weight_tf) + 1.0e-6)
        else:
            step_log_prob = tf.reduce_sum(selected_log_prob * source_active) / denom

        diag["mean_policy"] = tf.reduce_mean(probs, axis=0).numpy()
        self.prev_policy_probs = probs.numpy()
        return actions, {
            "log_prob": step_log_prob,
            "entropy": step_entropy,
            "kl_loss": step_kl_loss,
            "prior_loss": step_prior_loss,
            "value": step_value,
            "psi": psi,
            "diag": diag,
        }

    def train_episode(self, env, order_num_dist, reset_fn, seed: int, availability: float) -> Dict[str, float]:
        self.reset_episode()
        state = reset_fn(env, seed, availability)
        rewards = []
        log_probs = []
        entropies = []
        values = []
        psi_terms = []
        kl_losses = []
        prior_losses = []
        action_counts = []
        flow_drifts = []
        flow_abs = []
        total_reward = 0.0
        active_drivers = []

        with tf.GradientTape() as tape:
            for _ in range(144):
                drivers, _ = flatten_state(env, state)
                active_drivers.append(float(np.sum(drivers)))
                actions, step = self.select_action(env, state, order_num_dist, training=True)
                next_state, reward, _ = env.step(actions, 2)
                rewards.append(float(reward))
                total_reward += float(reward)
                log_probs.append(step["log_prob"])
                entropies.append(step["entropy"])
                values.append(step["value"])
                psi_terms.append(tf.reduce_mean(tf.abs(step["psi"])))
                kl_losses.append(step["kl_loss"])
                prior_losses.append(step["prior_loss"])
                action_counts.append(step["diag"]["num_actions"])
                flow_drifts.append(step["diag"]["flow_drift"])
                flow_abs.append(step["diag"]["flow_abs"])
                state = next_state

            returns = []
            running = 0.0
            for reward in reversed(rewards):
                running = reward + self.config.gamma * running
                returns.append(running)
            returns = list(reversed(returns))
            returns_tf = tf.convert_to_tensor(returns, dtype=tf.float32)
            returns_tf = (returns_tf - tf.reduce_mean(returns_tf)) / (tf.math.reduce_std(returns_tf) + 1.0e-6)
            values_tf = tf.stack(values)
            log_probs_tf = tf.stack(log_probs)
            entropies_tf = tf.stack(entropies)
            advantages = returns_tf - values_tf
            policy_loss = -tf.reduce_mean(log_probs_tf * tf.stop_gradient(advantages))
            value_loss = tf.reduce_mean(tf.square(advantages))
            entropy_loss = -tf.reduce_mean(entropies_tf)
            kl_loss = tf.reduce_mean(tf.stack(kl_losses))
            flow_loss = tf.reduce_mean(tf.stack(psi_terms))
            prior_loss = tf.reduce_mean(tf.stack(prior_losses))
            loss = (
                policy_loss
                + self.config.value_coef * value_loss
                + self.config.entropy_coef * entropy_loss
                + self.config.kl_coef * kl_loss
                + self.config.flow_coef * flow_loss
                + self.config.prior_coef * prior_loss
            )

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        day_orders = max(float(env.day_orders_num), 1.0)
        mean_active = max(float(np.mean(active_drivers)), 1.0)
        return {
            "train_orr": float(env.finished_order_num / day_orders),
            "train_adi": float(total_reward / mean_active),
            "reward": float(total_reward),
            "loss": float(loss.numpy()),
            "policy_loss": float(policy_loss.numpy()),
            "value_loss": float(value_loss.numpy()),
            "entropy": float(tf.reduce_mean(entropies_tf).numpy()),
            "kl_loss": float(kl_loss.numpy()),
            "flow_loss": float(flow_loss.numpy()),
            "prior_loss": float(prior_loss.numpy()),
            "grad_norm": float(grad_norm.numpy()),
            "flow_drift": float(np.mean(flow_drifts)),
            "flow_abs": float(np.mean(flow_abs)),
            "mean_actions": float(np.mean(action_counts)),
        }

    def eval_episode(self, env, order_num_dist, reset_fn, seed: int, availability: float) -> Dict[str, float]:
        self.reset_episode()
        state = reset_fn(env, seed, availability)
        total_reward = 0.0
        active_drivers = []
        action_counts = []
        flow_drifts = []
        flow_abs = []

        for _ in range(144):
            drivers, _ = flatten_state(env, state)
            active_drivers.append(float(np.sum(drivers)))
            actions, step = self.select_action(env, state, order_num_dist, training=False)
            state, reward, _ = env.step(actions, 2)
            total_reward += float(reward)
            action_counts.append(step["diag"]["num_actions"])
            flow_drifts.append(step["diag"]["flow_drift"])
            flow_abs.append(step["diag"]["flow_abs"])

        day_orders = max(float(env.day_orders_num), 1.0)
        mean_active = max(float(np.mean(active_drivers)), 1.0)
        return {
            "orr": float(env.finished_order_num / day_orders),
            "adi": float(total_reward / mean_active),
            "reward": float(total_reward),
            "mean_actions": float(np.mean(action_counts)),
            "flow_drift": float(np.mean(flow_drifts)),
            "flow_abs": float(np.mean(flow_abs)),
        }
