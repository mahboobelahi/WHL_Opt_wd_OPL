# Operational-layer diagnostics

## 1. Scope

The operational layer (OPL) is a deterministic post-optimization diagnostic for the fixed L1-L4 layouts used in Section 7 of the paper.

- OPL is not an NSGA-II or Beam Search objective.
- OPL does not alter or reselect structural layouts.
- OPL does not feed results back into the optimizer.
- The diagnostics are static access/assignment proxies, not routed warehouse simulation or calibrated travel time.

Locked pallet-slot capacities:

| Layout | Capacity |
|---|---:|
| L1 | 4,480 |
| L2 | 5,056 |
| L3 | 5,440 |
| L4 | 4,800 |

## 2. Checked-in reviewer evidence

Canonical fixed-layout evidence is under `data/operational_layer/paper_inputs/`:

- `selected_layouts.csv` — L1-L4 identity, provenance, and capacity.
- `slot_metrics_by_layout.csv` — all 19,776 pallet-slot rows with access distance, effective depth, vertical level, normalized descriptors, and slot cost.
- `sku_catalog.csv` — deterministic 100-SKU ABC catalog and Low=790 inventory.
- `representative_access_assignment.csv` — 400 representative assignments.
- `reserve_pallet_assignment.csv` — 2,760 canonical Low reserve assignments.
- `regime_A_metrics.csv` and `regime_B_metrics.csv`.
- `reserve_fragmentation_summary.csv`.
- `synthetic_orders.csv`, `order_effort_by_seed.csv`, `order_effort_summary.csv`.
- `sensitivity/lambda_sensitivity_summary.csv`, `sensitivity/lambda_sensitivity_by_seed.csv`.

Audited Round-2 occupancy evidence is under `data/operational_layer/occupancy_sensitivity/`. See its `README.md` and `source_result_inventory.csv`.

## 3. Occupancy protocol

The same L1-L4 layouts, slot geometry, 100 SKUs, ABC demand weights, SKU order, representative-assignment rule, and reserve-assignment rule are retained at every occupancy.

| Occupancy | Total pallets | Reserve pallets | Share of smallest capacity |
|---|---:|---:|---:|
| Low | 790 | 690 | 17.63% |
| Medium | 2,240 | 2,140 | 50.00% |
| High | 3,584 | 3,484 | 80.00% |

Inventory is scaled from the canonical 790-pallet vector by deterministic proportional largest remainder. Each SKU retains one representative-access pallet. Equal remainders are resolved by ascending `global_sku_index`.

Exact scaled quantities:

- Low: A `20 x 7`, B `30 x 15`, C `50 x 4`.
- Medium: A `20 x 20`, B `30 x 43`, C `50 x 11`.
- High: A `20 x 32`, B `30 x 68`, C `4 x 19` and `46 x 18`.

Representative assignments and Scenario-A diagnostics are occupancy-invariant. Reserve placement, utilization, reserve access, and reserve fragmentation may change.

## 4. Generate validated occupancy cases

Run from the repository root. The canonical source defaults to `data/operational_layer/paper_inputs`.

### Low reproduction gate

```powershell
python -m whl_experiments.build_operational_occupancy_sensitivity `
  --output-root results\section7_occupancy\low_790 `
  --occupancy-label low `
  --inventory-total 790 `
  --verify-low-against data\operational_layer\paper_inputs
```

### Medium

```powershell
python -m whl_experiments.build_operational_occupancy_sensitivity `
  --output-root results\section7_occupancy\medium_2240 `
  --occupancy-label medium `
  --inventory-total 2240
```

### High

```powershell
python -m whl_experiments.build_operational_occupancy_sensitivity `
  --output-root results\section7_occupancy\high_3584 `
  --occupancy-label high `
  --inventory-total 3584
```

Each generated case writes the scaled catalog, representative and reserve assignments, Scenario-A/B metrics, reserve fragmentation, validation summary, and occupancy manifest.

## 5. Cross-occupancy analysis

```powershell
python -m whl_experiments.analyze_operational_occupancy_sensitivity `
  --low-root results\section7_occupancy\low_790 `
  --medium-root results\section7_occupancy\medium_2240 `
  --high-root results\section7_occupancy\high_3584 `
  --output-root results\section7_occupancy\summary
```

Outputs:

- `occupancy_summary.csv`
- `occupancy_fragmentation_by_class.csv`
- `occupancy_reserve_components.csv`
- `occupancy_fragmentation_layout_summary.csv`
- `summary_manifest.json`

Reserve-access decomposition:

```text
mean(normalized_distance)
+ lambda_depth * mean(normalized_depth)
+ lambda_level * mean(normalized_level)
```

with baseline Scenario-B weights `(0.1, 0.1)`.

## 6. Weight sensitivity by occupancy

```powershell
python -m whl_experiments.analyze_operational_weight_sensitivity_by_occupancy `
  --low-root results\section7_occupancy\low_790 `
  --medium-root results\section7_occupancy\medium_2240 `
  --high-root results\section7_occupancy\high_3584 `
  --output-root results\section7_occupancy\weight_sensitivity
```

The canonical sensitivity files default to `data/operational_layer/paper_inputs/sensitivity/`.

| Case | depth weight | level weight |
|---|---:|---:|
| B1 | 0.10 | 0.10 |
| B2 | 0.25 | 0.10 |
| B3 | 0.10 | 0.25 |
| B4 | 0.25 | 0.25 |

The Low Scenario-B evidence must reproduce within absolute tolerance `1e-9` before Medium/High sensitivity is accepted.

## 7. Audited WHLR result transfer

The original WHLR `round2_section7_occupancy_sensitivity` result tree was audited before transfer. The source ZIP SHA-256 is:

`b3479f28d0e5d50a4e81b3ed012cdbf0d846a83e5eadcf0b94bda692c47b80fc`

Reviewer-facing transferred evidence now includes:

```text
data/operational_layer/occupancy_sensitivity/
  README.md
  scaled_inventory_summary.csv
  source_result_inventory.csv
  cases/
    low_790/
    medium_2240/
    high_3584/
  cross_occupancy/
  weight_sensitivity/
```

The three case validation summaries expose capacities, utilization, assignment row counts, class quantities, all scientific invariants, and the Low canonical reproduction gate. The case folders also expose the exact Scenario-B and class-specific fragmentation CSVs from WHLR. Cross-occupancy and weight-sensitivity CSVs are transferred directly from the validated result tree.

`source_result_inventory.csv` inventories all 33 original files with size, SHA-256, and CSV dimensions. Large occupancy-specific per-pallet assignment CSVs are not duplicated in the public data tree because they are deterministic generated outputs; their original hashes/row counts are recorded and the public wrapper regenerates them. Canonical Low raw assignments remain checked in under `paper_inputs/`.

## 8. Interpretation limits

The OPL evidence supports deterministic comparisons of fixed-layout access, reserve placement, fragmentation, utilization, and sensitivity to inventory load and proxy weights.

It must not be interpreted as routed travel time, throughput, congestion, batching, replenishment dynamics, service rate, or calibrated forklift performance. Synthetic-order effort aggregates the same fixed representative-access components under sampled SKU mixes; it is not an independent routing model.
