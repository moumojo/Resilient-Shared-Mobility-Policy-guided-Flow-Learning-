# Data interface

The ride-hailing records used in the paper are not redistributed. They contain
licensed or restricted trip and driver information. Users must obtain the
source datasets from their providers and apply the spatial projection,
ten-minute temporal aggregation, chronological split, and normalization used
in the paper.

The processed dispatch graphs contain 232 valid regions for CD and 255 valid
regions for NY. Training uses days 01--20 and testing uses days 21--30.

## Checkpoint inference

`code/run_test.py` accepts a NumPy graph archive containing:

```text
graph.npz
  action_targets  int32    [N, 7]
  valid_mask      bool/f32 [N, 7]
  adjacency       float32  [N, N]
```

`action_targets[i, a]` gives the destination node index for source node `i` and
action `a`. Actions 0--5 represent neighboring hexagonal movements and action 6
represents staying. `valid_mask` marks feasible action slots. Each adjacency row
must contain at least one positive entry and is normalized by the loader.

Node features are supplied as `features.npy` with shape `[N, F]` or `[T, N, F]`.
Released checkpoint inputs use the following feature order.

CD uses `F=17`:

```text
drivers, orders, future_demand, neighbor_drivers, neighbor_orders,
shortage, surplus, neighbor_shortage, supply_demand_ratio,
neighbor_supply_demand_ratio, gap, completion_potential, priority,
activity, time_sin, time_cos, peak
```

NY uses `F=18` and inserts `multi_hop_demand_potential` after
`neighbor_shortage`:

```text
drivers, orders, future_demand, neighbor_drivers, neighbor_orders,
shortage, surplus, neighbor_shortage, multi_hop_demand_potential,
supply_demand_ratio, neighbor_supply_demand_ratio, gap,
completion_potential, priority, activity, time_sin, time_cos, peak
```

Count-like checkpoint features are divided by their regional 90th percentile
of absolute magnitude plus one. The generated graph used by `--verify-all`
checks checkpoint compatibility only and cannot reproduce paper metrics.

The full five-layer PGFA training implementation constructs its own
20-dimensional enhanced state in `code/pgfa_full.py`; this state is separate
from the 17/18-dimensional released-checkpoint interface.

## CityReal simulator inputs

A compatible CD data directory contains:

```text
map_232.npy
small_order_num_dist_day_01_20.npy
small_order_real_day_01_20.npy
small_order_num_dist_day_21_30.npy
small_order_real_day_21_30.npy
idle_driver_dist_time.npy
idle_driver_location_mat_232.npy
onoff_driver_location_mat_232.npy
```

A compatible NY data directory contains:

```text
map_nyc.npy
nyc_order_num_dist_train.npy
nyc_order_real_train.npy
nyc_order_num_dist_test.npy
nyc_order_real_test.npy
idle_driver_dist_time.npy
idle_driver_location_mat_nyc.npy
onoff_driver_location_mat_nyc.npy
```

`idle_driver_dist_time.npy` may be initialized to zeros when that signal is not
available. `map_*.npy` stores the two-dimensional hexagonal grid map.
`order_num_dist_*` stores regional demand distributions over 144 ten-minute
steps per day. `order_real_*` stores simulator-ready orders.
`idle_driver_location_mat_*` initializes regional idle supply, and
`onoff_driver_location_mat_*` stores time-dependent driver online/offline
changes.

## Data safety

The loaders use NumPy object arrays for simulator-ready order records. Only load
`.npy` or `.npz` files obtained from a trusted source. Do not commit raw trip,
driver, location, credential, or personally identifiable data to this
repository.
