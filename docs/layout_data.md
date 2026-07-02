***TOC:***
- [Layout Data](#layout-data)
  - [1. Layout categories](#1-layout-categories)
    - [1. Demo/example optimization layouts](#1-demoexample-optimization-layouts)
    - [2. Optimization benchmark layouts](#2-optimization-benchmark-layouts)
    - [3. Reference/comparison-only layouts](#3-referencecomparison-only-layouts)
  - [4. Important usage rules](#4-important-usage-rules)
  - [5. Editor and plotting use](#5-editor-and-plotting-use)
---

# Layout Data

Layout masks are stored under:

```text
data/instances/masks/
```

These `.npz` files are warehouse layout or mask inputs generated from Tkinter editor app. They are not generated optimization outputs. Generated results are written under `results/`, which is ignored by Git.

## 1. Layout categories

The layout masks are grouped by use.

### 1. Demo/example optimization layouts

These layouts are intended for quick checks and examples. They may be used by the optimization runners.

- `demo_layout_door_bottom_AW_2.npz`
- `demo_layout_door_bottom_AW_3.npz`
- `demo_layout_door_left_AW_2.npz`
- `demo_layout_door_left_AW_3.npz`
- `demo_layout_door_UB_AW_2.npz`
- `demo_layout_door_UB_AW_3.npz`

### 2. Optimization benchmark layouts

These layouts may be used by the optimization runners.

- `Gyorgy-KOVACS_MWH_Narrow_AW_4.npz`
- `Gyorgy-KOVACS_MWH_Wide_AW_5.npz`
- `Gyorgy-KOVACS_WH_Narrow_AW_4.npz`
- `Gyorgy-KOVACS_WH_Wide_AW_5.npz`
- `Answer_Set_layout_AW_1.npz`
- `Answer_Set_layout_AW_2.npz`
- `Answer_Set_layout_AW_3.npz`
- `AT_S_comercial_layout_AW_3.npz`

### 3. Reference/comparison-only layouts

These layouts are not used in default optimization discovery:

- `AT_1.npz` through `AT_13.npz`

They remain available for editor preview, plotting, and comparison workflows.

## 4. Important usage rules

`AT_S_comercial_layout_AW_3.npz` is an optimization layout and remains included in default optimization discovery.

`AT_1.npz` through `AT_13.npz` are reference/comparison-only layouts. They should not be treated as optimization instances.

For command-line instance selection and accepted parameters, see `docs/cli_commands.md`.

## 5. Editor and plotting use

The Tkinter editor uses the original grid preview behavior for inspecting and editing masks.

Paper-style Pareto plotting is separate from the editor preview and reads CSV inputs from:

```text
data/plot_inputs/paper/
```

The required paper Pareto plotting inputs are documented in `docs/benchmark_sources.md` and `docs/cli_commands.md`.
