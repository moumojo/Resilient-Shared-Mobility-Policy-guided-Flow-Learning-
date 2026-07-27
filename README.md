# Policy-Guided Spatiotemporal Graph Learning: Direction-Aware Information Fusion for Urban Mobility Resource Allocation

Official model, simulator, and checkpoint release for **Policy-Guided Spatiotemporal Graph Learning: Direction-Aware Information Fusion for Urban Mobility Resource Allocation**.

PGFA combines a Policy Flow Prior, Policy-guided Message passing, Flow-aware
Attention, and actor--critic reinforcement learning for city-scale
ride-hailing fleet rebalancing. The release includes the full PGFA training
components, the CityReal simulator, pretrained checkpoints used for the paper
tables, four ablation families, and seven paper-inspired comparison models.

## Repository contents

```text
PGFA/
  code/
    pgfa_full.py          PGFA states, graph modules, actor--critic losses,
                          execution projection, and checkpoint architectures
    baseline_models.py    Seven comparison-model architectures
    run_test.py           Checkpoint discovery, verification, and inference
    load_weights.py       Compact PGFA checkpoint loader
    build_manifest.py     Checkpoint manifest generator
  simulator/              CityReal environment, entities, and utilities
  result/                 76 paper and ablation/comparison checkpoints
  weights/                Compact six-checkpoint PGFA subset with metrics
  CHECKPOINTS.csv         SHA-256 release manifest
  DATA.md                 Data, graph, and feature interfaces
  environment.yml         Recommended Conda environment
  requirements.txt        Pinned Python dependencies
  LICENSE                 GNU GPL v3
```

The processed dispatch graphs contain 232 valid regions for CD (Chengdu) and
255 valid regions for NY (New York). Each policy has seven feasible-action
slots: six adjacent hexagonal movements and one stay action.

## Installation

The compatibility environment is Python 3.8, TensorFlow 2.5.0, and NumPy
1.19.5.

```bash
conda env create -f environment.yml
conda activate pgfa
```

Alternatively, use an existing Python 3.8 environment:

```bash
python -m pip install -r requirements.txt
```

Run commands from the repository root shown above.

## Verify the release

List every released checkpoint without loading TensorFlow variables:

```bash
python code/run_test.py --list
```

Build every corresponding architecture, strictly restore its model variables,
and run a forward pass:

```bash
python code/run_test.py --verify-all
```

A successful release reports:

```json
{
  "total": 76,
  "passed": 76,
  "failed": 0
}
```

Regenerate `CHECKPOINTS.csv` and verify its hashes:

```bash
python code/build_manifest.py
python code/build_manifest.py --check
```

## Load a checkpoint

Use an exact checkpoint ID printed by `--list`, or a unique substring:

```bash
python code/run_test.py \
  --checkpoint CD/results_pgfa_rl_100
```

Without external inputs, this command uses zero-valued features and a generated
compatibility graph. It verifies restoration only and does not reproduce paper
metrics.

For inference with processed inputs:

```bash
python code/run_test.py \
  --checkpoint NY/results_pgfa_rl_nyc_66 \
  --graph-npz path/to/graph.npz \
  --features-npy path/to/features.npy \
  --output-dir outputs/ny_pgfa_66
```

The output directory contains `logits.npy`, `action_probs.npy`, `actions.npy`,
`values.npy`, `aux.npy`, and `summary.json`. For PGFA checkpoints, `aux.npy`
stores the learned node propensity values.

## Released models

The compact `weights/` tree contains the six PGFA checkpoints corresponding to
the two cities and three vehicle-availability settings. `result/` additionally
contains the four ablation families and these seven comparison families:

- BloomSignature-GNN
- OGIF-GAT
- MultiModal-STGCN
- WLA-STGCN
- MM-STMAP
- DT-D3QN
- QoSQueue-DTSM

`pgfa_full.py` contains the full five-layer state encoder and two-layer PGFA
actor--critic implementation. It also contains the checkpoint-compatible model
needed to restore the released paper weights. The two architectures use
separate constructors and their weights are not interchangeable.

## Simulator and data

The simulator is available from the package root:

```python
from simulator import CityReal
```

`CityReal` executes order generation, driver state transitions, neighboring
grid movement, order matching, and environment statistics over 144 ten-minute
decision steps.

Raw and processed ride-hailing records are not redistributed because their
licenses or access conditions do not permit inclusion in this repository.
`DATA.md` defines the required filenames, graph arrays, feature order, and
chronological split.

## Reproducibility scope

This public release supports architecture inspection, strict restoration of all
released weights, forward inference on compatible processed inputs, and use of
the included simulator. Exact regeneration of ORR and normalized ADI requires
the restricted city records, their documented preprocessing, the real city
graphs, and chronological test snapshots. Generated compatibility graphs and
zero-valued features are never substitutes for the evaluation data.

The comparison architectures are included for checkpoint inspection and
inference. Their historical experiment runner depended on private workspace
modules and is not presented as a standalone retraining command in this
release.

## License and contact

The code is released under the GNU General Public License v3.0. Dataset terms
remain governed by their original providers and are not changed by this
software license.

For research questions and reproducibility reports, contact Xiaohui Huang at
`huangxiaohui@ecjtu.edu.cn`.
