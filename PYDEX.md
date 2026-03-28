# PYDEX — Project Reference

## Overview

**pydex** (Python Design of Experiments) is an open-source package for **Optimal Experiment Design (OED)**. It helps researchers determine which experiments to run to maximally inform parameter estimation of mathematical models.

- Version: 0.0.9
- Author: Kennedy Putra Kusumo
- License: MIT
- Python: >= 3.6
- PyPI: `pip install pydex`

---

## Repository Structure

```
pydex/
├── pydex/
│   ├── core/
│   │   ├── designer.py          # Main class — 5,390 lines, ~160 methods
│   │   ├── logger.py            # Dual-stream logger (console + file)
│   │   └── bnb/
│   │       ├── node.py          # Branch-and-bound node (146 lines)
│   │       └── tree.py          # Branch-and-bound tree (105 lines)
│   └── utils/
│       ├── trellis_plotter.py   # High-dimensional visualization
│       └── dynamic_experiment_plotter.py  # Dynamic system visualization
├── examples/                    # 17 subdirectories, 50+ scripts
│   ├── linear/                  # Polynomial models (order 1–7)
│   ├── non_linear/              # Exponential and non-linear models
│   ├── ode/                     # ODE-based models (pyomo/scipy)
│   ├── estimability/            # Estimability analysis
│   ├── exact_experiments/       # Discrete design / branch-and-bound
│   ├── cvar/                    # CVaR robust design
│   ├── dynamic/                 # Time-varying controls
│   ├── pseudo_bayesian/         # Bayesian-averaged designs
│   ├── prediction_oriented/     # V-optimality / prediction criteria
│   ├── time_varying_controls/   # Dynamic control strategies
│   ├── non_cubic_spaces/        # Non-rectangular feasible regions
│   ├── parameter_estimation/    # Parameter estimation examples
│   └── *.ipynb                  # Quickstart and ODE notebooks
├── run_tests.py                 # Integration test runner
├── setup.py
└── pyproject.toml
```

---

## Core Architecture

### Designer Class (`pydex/core/designer.py`)

The single user-facing class. Users subclass or monkey-patch the `simulate()` method.

**Key responsibilities:**
- Manage experimental candidates (TIC, TVC, SPT)
- Compute sensitivities (numdifftools, adaptive finite difference + Richardson extrapolation)
- Build Fisher Information Matrices (FIM)
- Optimize experiment designs (scipy / cvxpy)
- Estimate parameters (least-squares, MCMC)
- Visualize results (20+ matplotlib plotting methods)

**State management:** 100+ instance attributes with dirty flags (`_candidates_changed`, `_model_parameters_changed`, etc.) to avoid redundant recomputation.

**Typical workflow:**
```python
from pydex.core.designer import Designer
import numpy as np

def simulate(ti_controls, model_parameters):
    return np.array([...])  # returns response vector

d = Designer()
d.simulate = simulate
d.model_parameters = [1.0, 2.0]
d.ti_controls_candidates = np.linspace(-1, 1, 11)[:, None]
d.initialize()
d.design_experiment(d.d_opt_criterion)
d.apportion(n_exp=5)
d.plot_optimal_efforts()
```

### Branch-and-Bound (`pydex/core/bnb/`)

Used for exact (discrete/integer) designs.
- `Node`: holds a cvxpy subproblem, tracks bounds, checks integrality
- `Tree`: manages node pool, iterates, determines solution
- Branching strategy: "greatest fractional" variable

---

## Design Criteria

### Parameter Estimation-Oriented
| Criterion | Goal |
|-----------|------|
| D-optimality | Maximize log-det(FIM) — minimize parameter uncertainty volume |
| A-optimality | Minimize trace(FIM⁻¹) — minimize average parameter variance |
| E-optimality | Maximize min eigenvalue(FIM) — minimize worst-case variance |

### Prediction-Oriented
- `dg_opt`, `di_opt`, `ag_opt`, `ai_opt`, `eg_opt`, `ei_opt` — analogs of D/A/E for prediction rather than estimation

### Robust / Risk-Averse
- `cvar_d_opt_criterion` — CVaR-based D-optimality (robust to parameter uncertainty)
- Pseudo-Bayesian: average FIM over a parameter distribution

---

## Experiment Specification

| Component | Code name | Description |
|-----------|-----------|-------------|
| Time-Invariant Controls | `ti_controls_candidates` | Constants during experiment (e.g., temperature) |
| Time-Varying Controls | `tv_controls_candidates` | Dynamic inputs (e.g., feed rate over time) |
| Sampling Times | `sampling_times_candidates` | When measurements are taken (can be optimized) |
| Candidates | Auto-generated | All combinations of TIC/TVC/SPT |
| Efforts | `efforts` | Continuous weights per candidate (sum to 1) |

---

## Key Dependencies

| Package | Role |
|---------|------|
| `numpy` | Array operations |
| `scipy` | Optimization, statistics |
| `matplotlib` | All visualization |
| `numdifftools` | Numerical Jacobians (Richardson extrapolation) |
| `cvxpy` | Convex optimization interface |
| `dill` | Pickle with weakref support |
| `emcee` | MCMC for Bayesian parameter estimation |
| `corner` | Corner plots for posterior visualization |

---

## Testing

Integration tests in `run_tests.py` — runs full example scripts:
- `examples.non_linear.test_non_linear`
- `examples.linear.test_linear_examples`
- `examples.ode.test_ode`
- `examples.non_cubic_spaces.test_non_cubic_spaces`
- `examples.time_varying_controls.test_tvc`

No unit tests currently exist for individual methods.

---

## Code Style Notes

- Python 3.6-era style: no type hints, no dataclasses, `os.path` patterns
- Large monolithic class (`designer.py`) — 5,390 lines, ~160 methods
- Factory pattern for criteria methods
- Strategy pattern for solver selection (scipy vs cvxpy)
- Template method: `simulate()` is overridden by users
- Extensive matplotlib-based visualization (not modular)
- Mix of numpy and pure-Python loops in some places

---

## Development Notes

- Version 0.0.9 — active development, API not yet stable
- No CI/CD pipeline currently
- Documentation structure exists (`docs/readthedocs_structure.docx`) but ReadTheDocs not set up
- TODO list in `pydex_todo.docx`
