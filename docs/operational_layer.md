### Operational-layer diagnostics

***TOC:***
- [1. Purpose](#1-purpose)
- [2. Scope](#2-scope)
- [3. Inputs](#3-inputs)
- [4. Workflow](#4-workflow)
- [5. Outputs](#5-outputs)
- [6. Commands](#6-commands)
- [7. Limitations](#7-limitations)
-----

## 1. Purpose

The operational-layer diagnostics are an optional post-optimization analysis layer for selected representative warehouse layouts. They package the L1-L4 layouts used in the revised manuscript and compute fixed proxy diagnostics over those layouts.

This layer is intended for **reviewer traceability and manuscript reproduction**. It is not a replacement for the structural optimizer, and it is not a routed warehouse simulation.

## 2. Scope

OPL is separate from structural optimization:

- OPL is not part of NSGA-II fitness evaluation.
- OPL does not feed back into the optimizer.
- OPL is applied after Pareto-front screening and representative-layout selection.
- In this release, OPL is limited to the L1-L4 representative layouts used in the paper.

The restored scripts are fixed-path reproduction helpers. They use the checked-in OPL data files under `data/operational_layer/`.

## 3. Inputs

Paper-used OPL inputs are stored under `data/operational_layer/`.

`data/operational_layer/config/` contains the fixed diagnostic assumptions:

- `operational_config.json`

`data/operational_layer/paper_inputs/` contains the selected-layout and diagnostic input tables:

- `selected_layouts.csv`
- `candidate_layout_screening.csv`
- `unique38_layout_review_table.csv`
- `slot_metrics_by_layout.csv`
- `sku_catalog.csv`
- `representative_access_assignment.csv`
- `reserve_pallet_assignment.csv`
- `regime_A_metrics.csv`
- `regime_B_metrics.csv`
- `reserve_fragmentation_summary.csv`
- `synthetic_orders.csv`
- `order_effort_by_seed.csv`
- `order_effort_summary.csv`
- `sensitivity/lambda_sensitivity_summary.csv`
- `sensitivity/lambda_sensitivity_by_seed.csv`

The selected L1-L4 panel and supporting plots are under `data/operational_layer/layout_panel/`.

## 4. Workflow

The paper-used workflow is:

1. Select representative layouts L1-L4 from structural optimization results.
2. Extract slot/storage metrics for the fixed selected layouts.
3. Generate the deterministic SKU catalog.
4. Assign representative-access and reserve pallet assumptions.
5. Compute Scenario A diagnostics.
6. Compute Scenario B diagnostics.
7. Compute the fixed synthetic order proxy.
8. Compute lambda-sensitivity summaries for appendix/supporting evidence.
9. Package manuscript-ready tables and figures.

Some restored source files still use older internal names such as `regime_A` and `regime_B`; these correspond to Scenario A and Scenario B in the documentation.

## 5. Outputs

`data/operational_layer/paper_outputs/logs/` contains JSON summaries for the final workflow:

- `m3C_final_selection_lock_summary.json`
- `m4_slot_metrics_summary.json`
- `m5_sku_catalog_summary.json`
- `m6_assignment_summary.json`
- `m7_regime_metrics_summary.json`
- `m8_order_proxy_summary.json`
- `m9_lambda_sensitivity_summary.json`
- `m10_manuscript_outputs_summary.json`

`data/operational_layer/paper_outputs/manuscript/` contains manuscript-ready tabular outputs:

- `table_operational_diagnostic.csv`
- `table_operational_diagnostic.tex`

`data/operational_layer/layout_panel/` contains the selected-layout panel and diagnostic plot artifacts:

- `selected_layouts_panel_final_L1_L4.png`
- `20Figure_12_selected_layouts_panel_final_L1_L4.png`
- `selected_layouts_panel.png`
- `regime_score_barplot.png`

## 6. Commands

The restored OPL files do not expose public argparse commands. They are internal/reproduction helpers with fixed project-relative paths:

```powershell
python -m whl_experiments.extract_operational_slot_metrics
python -m whl_experiments.generate_operational_sku_catalog
python -m whl_experiments.assign_operational_sku_pallets
python -m whl_experiments.compute_operational_regime_metrics
python -m whl_experiments.compute_synthetic_order_proxy
python -m whl_experiments.compute_operational_lambda_sensitivity
python -m whl_experiments.prepare_operational_manuscript_outputs
```

Run them only when intentionally regenerating the OPL diagnostic artifacts. They are not part of normal optimization runs and should not be used as optimizer validation commands.

## 7. Limitations

OPL is a diagnostic proxy layer. It is not a routed picker simulation, not throughput validation, and not a feedback objective for the structural optimizer.

The release is intentionally limited to the selected paper layouts and assumptions. Broad campaign screening scripts and older selection prototypes are not included in the public OPL workflow.