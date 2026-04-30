# pydex — Python Design of Experiments

An open-source Python package for optimal experiment design, essential to
a modeller's toolbelt. If you develop a model of any kind, you will relate
to the challenges of estimating its parameters. This tool helps design
maximally informative experiments for collecting data to calibrate your
model.

## Fork Notice

This is a fork of [pydex](https://github.com/KennedyPutraKusumo/pydex)
by Kennedy Putra Kusumo et al., originally described in:

> Kusumo, K.P., Kuriyan, K., Vaidyanathan, R., Shadbahr, T., Khan, F.I.,
> Harun, N., Sin, G. & Braatz, R.D. (2022). Explicit-Constraint-Based
> Optimal Experiment Design for Differential Equation Models.
> *Computers & Chemical Engineering*, 107940.

### Changes in this fork

- **V-optimal MBDoE**: two-stage workflow targeting prediction accuracy
  at a user-specified operating condition (`find_optimal_operating_point()`
  \+ `design_v_optimal()`)
- **Process optimisation**: constrained nonlinear optimisation via Pyomo
  to find the optimal operating point (Stage 1) before designing experiments
- **Pyomo-centric solver**: cvxpy removed; the pydex OED problem is now
  formulated and solved entirely via Pyomo, giving access to any solver
  callable through Pyomo (`solver="ipopt"`, `solver="gams"`, etc.) using
  `design_experiment(solver=..., solver_options={...})`
- **Parallel IFT sensitivity evaluation**: when a `pyomo_model_fn` is
  provided, sensitivities are computed via the Implicit Function Theorem
  (IFT) using the Pyomo NLP Jacobian; parallelised over candidates (local
  designs) or scenarios (pseudo-Bayesian designs) via `joblib` loky workers
- **Pyomo.DAE support**: full DAE models can serve as both the simulator
  and the IFT sensitivity source; signature-2 multi-output models
  supported
- **regularize_fim fix**: `regularize_fim=True` now correctly adds ε·I
  to the symbolic FIM expression in the Pyomo solve (previously the flag
  was stored but had no computational effect on the native solve path)
- **Comprehensive test suite**: 32 end-to-end tests covering all design
  criteria, both sensitivity paths (FD and IFT), parallel correctness,
  prior FIM, save/load, visualisation, and more
- **Improved documentation**: Sphinx/NumPy-compliant docstrings throughout
  `designer.py`

---

## Installation

Install directly from this fork:

```bash
pip install git+https://github.com/salvadorgarciamunoz/pydex.git
```

---

## Quick Start

See the `examples/` folder for worked examples using the current API.
A good starting point is `examples/first_order_reaction_ipopt.py` for a
simple D-optimal design, or `examples/v_optimal_test_case.py` for a
full two-stage V-optimal workflow.

---

## Features

1. Simple, intuitive syntax — easy to get started, powerful enough for
   complex problems.
2. Continuous and exact (discrete) experimental designs via Adams
   apportionment.
3. Design criteria: D-optimal, A-optimal, E-optimal, V-optimal, CVaR
   variants, pseudo-Bayesian (types 0 and 1), MINLP sparse designs.
4. OED problem formulated entirely in Pyomo — any solver accessible
   through Pyomo can be used.
5. Pyomo.DAE support: DAE models as simulator and IFT sensitivity source.
6. Parallel sensitivity evaluation via `joblib` loky workers.
7. Convenient built-in visualisation via matplotlib.
8. Supports virtually any model written as a Python function, including
   ODE models solved via scipy or Pyomo.DAE.

---

## Dependencies

### Core (installed automatically)

| Package        | Purpose                                              |
|----------------|------------------------------------------------------|
| numpy          | Array operations                                     |
| scipy          | ODE integration, optimisation fallback               |
| matplotlib     | Visualisation                                        |
| numdifftools   | Numerical finite-difference sensitivities            |
| pyomo          | OED problem formulation, DAE modelling, IFT Jacobian |
| joblib         | Parallel sensitivity evaluation                      |
| dill           | Saving objects with weak references                  |
| corner         | Corner plots for high-dimensional parameter spaces   |
| emcee          | Bayesian MCMC inference                              |

### Solvers

The pydex OED problem is formulated entirely in Pyomo, so solver
requirements depend on the type of problem being solved:

**Continuous designs** (standard D/A/E/V-optimal, pseudo-Bayesian, CVaR)
require a nonlinear programming (NLP) solver accessible through Pyomo.
IPOPT is recommended:

```bash
pip install cyipopt
```

> **Note:** `cyipopt` provides a Python interface to IPOPT. For best
> performance, configure IPOPT with HSL linear solvers (`MA27`, `MA57`)
> — see `docs/ipopt_setup_guide.docx` for installation instructions.
> The open-source solver `MUMPS` is available as a fallback.

Any other NLP solver supported by Pyomo (e.g. `glpk`, `cplex`) can also
be used by passing `solver=<solver_name>` to `design_experiment()`.

**Sparsity-enforcing MINLP designs** (`min_effort > 0`) require a
mixed-integer nonlinear programming (MINLP) solver. BARON via GAMS is
recommended:

```python
designer.design_experiment(
    criterion  = designer.d_opt_criterion,
    solver     = "gams",
    min_effort = 0.05,
)
```

---

## API Overview

### Basic D-optimal design

```python
from pydex.core.designer import Designer
import numpy as np

designer = Designer()
designer.simulate               = my_simulate_fn   # callable(tic, mp) -> array
designer.model_parameters       = np.array([...])
designer.ti_controls_candidates = candidate_grid

designer.initialize(verbose=1)
designer.eval_sensitivities()

designer.design_experiment(
    criterion      = designer.d_opt_criterion,
    solver         = "ipopt",
    solver_options = {"linear_solver": "ma57", "tol": 1e-8},
)
designer.print_optimal_candidates()
designer.apportion(n_exp=10)
```

### Pyomo.DAE model with automatic IFT and parallelisation

When `pyomo_model_fn` is provided, `use_pyomo_ift` and `n_jobs` are
auto-set at `initialize()` — no manual configuration needed:

```python
designer.simulate         = my_simulate_fn      # for predictions
designer.pyomo_model_fn   = my_build_model_fn   # for IFT Jacobian
designer.model_parameters = np.array([...])
designer.initialize(verbose=1)
# use_pyomo_ift=True and n_jobs=-1 set automatically

designer.design_experiment(
    criterion            = designer.d_opt_criterion,
    solver               = "ipopt",
    solver_options       = {"linear_solver": "ma57"},
    pseudo_bayesian_type = 0,   # for pseudo-Bayesian designs
)
```

### Pseudo-Bayesian design

```python
scenarios = np.column_stack([
    np.random.uniform(lb, ub, N),  # one column per uncertain parameter
    ...
])
designer.model_parameters = scenarios   # shape (N_scenarios, n_mp)

designer.design_experiment(
    criterion            = designer.d_opt_criterion,
    solver               = "ipopt",
    solver_options       = {"linear_solver": "ma57"},
    pseudo_bayesian_type = 0,   # 0 = average FIM; 1 = average criterion
)
```

---

## V-optimal MBDoE

V-optimal design minimises model prediction variance at a specific
operating condition `dw` (e.g. the economically optimal process point),
rather than minimising global parameter uncertainty as D/A-optimal designs
do. It follows a two-stage workflow:

**Stage 1 — Process optimisation:** find `dw` by solving a constrained
nonlinear programme over the operating space via Pyomo.

**Stage 2 — V-optimal MBDoE:** design experiments that minimise
`J_V = trace(W FIM⁻¹ Wᵀ)` where `W` is the scaled sensitivity matrix
evaluated at `dw`.

```python
# Stage 1
designer.process_objective   = my_objective    # callable(tic, tvc, mp) -> float
designer.process_constraints = my_constraints  # callable(tic, tvc, mp) -> list
designer.dw_sense            = "maximize"
designer.dw_bounds_tic       = [(lb, ub), ...]

designer.find_optimal_operating_point(
    init_guess     = np.array([[60.0, 70.0, 1.0]]),
    solver         = "ipopt",
    solver_options = {"linear_solver": "ma57"},
)

# Stage 2
designer.dw_spt = np.array([t_final])
designer.design_experiment(
    criterion               = designer.v_opt_criterion,
    solver                  = "ipopt",
    solver_options          = {"linear_solver": "ma57"},
    optimize_sampling_times = True,
)
```

See `examples/v_optimal_test_case.py` for a complete worked example with
a batch reactor system featuring three competing reactions.

> Shahmohammadi, A. & McAuley, K.B. (2019). Sequential model-based A- and
> V-optimal design of experiments for building fundamental models of
> pharmaceutical production processes. *Computers & Chemical Engineering*,
> 129, 106504. https://doi.org/10.1016/j.compchemeng.2019.06.029

---

## Examples

The `examples/` folder contains worked examples:

- `examples/v_optimal_test_case.py` — V-optimal MBDoE, batch reactor,
  three competing reactions, two-stage workflow
- `examples/first_order_reaction_ipopt.py` — D-optimal design with IPOPT,
  first-order ODE, finite-difference sensitivities
- `examples/first_order_reaction_pyomo.py` — D-optimal design with
  Pyomo.DAE and IFT sensitivities, parallel evaluation
- `examples/ivt_mbdoe.py` — industrial IVT (In-Vitro Testing) MBDoE
  application with batch reactor model

---

## Running the Tests

```bash
python pydex_full_capability_test.py
```

32 end-to-end tests covering all design criteria, both sensitivity paths
(finite-difference and Pyomo IFT), sequential vs parallel correctness,
prior FIM, save/load, and visualisation.

---

Do you have a question, suggestion, or feature request? Feel free to open
an issue or contact the original author at
[kennedy.putra.kusumo@gmail.com](mailto:kennedy.putra.kusumo@gmail.com).


An open-source Python package for optimal experiment design, essential to
a modeller's toolbelt. If you develop a model of any kind, you will relate
to the challenges of estimating its parameters. This tool helps design
maximally informative experiments for collecting data to calibrate your
model.

## Fork Notice

This is a fork of [pydex](https://github.com/KennedyPutraKusumo/pydex)
by Kennedy Putra Kusumo et al., originally described in:

> Kusumo, K.P., Kuriyan, K., Vaidyanathan, R., Shadbahr, T., Khan, F.I.,
> Harun, N., Sin, G. & Braatz, R.D. (2022). Explicit-Constraint-Based
> Optimal Experiment Design for Differential Equation Models.
> *Computers & Chemical Engineering*, 107940.

### Changes in this fork

- **V-optimal MBDoE**: two-stage workflow targeting prediction accuracy
  at a user-specified operating condition (`find_optimal_operating_point()`
  \+ `design_v_optimal()`)
- **Process optimisation**: constrained nonlinear optimisation via IPOPT
  to find the optimal operating point (Stage 1) before designing experiments
- **Pyomo/IPOPT-centric solver**: cvxpy removed; all design criteria
  (D/A/E/V-optimal, CVaR, pseudo-Bayesian) now use IPOPT via `cyipopt`
  with `design_experiment(solver="ipopt", solver_options={...})`
- **Parallel IFT sensitivity evaluation**: when a `pyomo_model_fn` is
  provided, sensitivities are computed via the Implicit Function Theorem
  (IFT) using the Pyomo NLP Jacobian; parallelised over candidates (local
  designs) or scenarios (pseudo-Bayesian designs) via `joblib` loky workers
- **Pyomo.DAE support**: full DAE models can serve as both the simulator
  and the IFT sensitivity source; signature-2 multi-output models
  supported
- **regularize_fim fix**: `regularize_fim=True` now correctly adds ε·I
  to the symbolic FIM expression in the IPOPT solve (previously the flag
  was stored but had no computational effect on the native solve path)
- **Comprehensive test suite**: 32 end-to-end tests covering all design
  criteria, both sensitivity paths (FD and IFT), parallel correctness,
  prior FIM, save/load, visualisation, and more
- **Improved documentation**: Sphinx/NumPy-compliant docstrings throughout
  `designer.py`

---

## Installation

Install directly from this fork:

```bash
pip install git+https://github.com/salvadorgarciamunoz/pydex.git
```

Or install the original release from PyPI:

```bash
pip install pydex
```

---

## REPLACE_ME

- [Demo](https://github.com/KennedyPutraKusumo/pydex/blob/master/examples/pydex_quickstart.ipynb)
  of basic features.
- [Example](https://github.com/KennedyPutraKusumo/pydex/blob/master/examples/pydex_ode_model.ipynb)
  of experimental design for ODE models.
- [V-optimal example](examples/v_optimal_test_case.py): two-stage
  prediction-oriented MBDoE for a batch reactor with competing reactions.

---

## Features

1. Simple, intuitive syntax — easy to get started, powerful enough for
   complex problems.
2. Continuous and exact (discrete) experimental designs via Adams
   apportionment.
3. Design criteria: D-optimal, A-optimal, E-optimal, V-optimal, CVaR
   variants, pseudo-Bayesian (types 0 and 1), MINLP sparse designs.
4. IPOPT (via cyipopt) as the primary solver for all criteria.
5. Pyomo.DAE support: DAE models as simulator and IFT sensitivity source.
6. Parallel sensitivity evaluation via `joblib` loky workers.
7. Convenient built-in visualisation via matplotlib.
8. Supports virtually any model written as a Python function, including
   ODE models solved via scipy or Pyomo.DAE.

---

## Dependencies

### Core (installed automatically)

| Package        | Purpose                                              |
|----------------|------------------------------------------------------|
| numpy          | Array operations                                     |
| scipy          | ODE integration, optimisation fallback               |
| matplotlib     | Visualisation                                        |
| numdifftools   | Numerical finite-difference sensitivities            |
| pyomo          | DAE modelling and IFT Jacobian extraction            |
| joblib         | Parallel sensitivity evaluation                      |
| dill           | Saving objects with weak references                  |
| corner         | Corner plots for high-dimensional parameter spaces   |
| emcee          | Bayesian MCMC inference                              |

### Required: IPOPT

IPOPT is the primary solver for all design criteria in this fork.

```bash
pip install cyipopt
```

> **Note:** `cyipopt` requires a working IPOPT installation with a
> compatible linear solver. The open-source solver `MUMPS` is bundled
> with most binary distributions. For best performance, HSL solvers
> (`MA27`, `MA57`) are recommended — see
> `docs/ipopt_setup_guide.docx` for installation instructions.

### Optional: GAMS + BARON

Required only for sparsity-enforcing MINLP designs
(`design_experiment(solver="gams", min_effort=0.05)`).

---

## API Overview

### Basic D-optimal design

```python
from pydex.core.designer import Designer
import numpy as np

designer = Designer()
designer.simulate             = my_simulate_fn   # callable(tic, mp) -> array
designer.model_parameters     = np.array([...])
designer.ti_controls_candidates = candidate_grid

designer.initialize(verbose=1)
designer.eval_sensitivities()

designer.design_experiment(
    criterion      = designer.d_opt_criterion,
    solver         = "ipopt",
    solver_options = {"linear_solver": "ma57", "tol": 1e-8},
)
designer.print_optimal_candidates()
designer.apportion(n_exp=10)
```

### Pyomo.DAE model with automatic IFT and parallelisation

When `pyomo_model_fn` is provided, `use_pyomo_ift` and `n_jobs` are
auto-set at `initialize()` — no manual configuration needed:

```python
designer.simulate         = my_simulate_fn      # for predictions
designer.pyomo_model_fn   = my_build_model_fn   # for IFT Jacobian
designer.model_parameters = np.array([...])
designer.initialize(verbose=1)
# use_pyomo_ift=True and n_jobs=-1 set automatically

designer.design_experiment(
    criterion            = designer.d_opt_criterion,
    solver               = "ipopt",
    solver_options       = {"linear_solver": "ma57"},
    pseudo_bayesian_type = 0,   # for pseudo-Bayesian designs
)
```

### Pseudo-Bayesian design

```python
scenarios = np.column_stack([
    np.random.uniform(lb, ub, N),  # one column per uncertain parameter
    ...
])
designer.model_parameters = scenarios   # shape (N_scenarios, n_mp)

designer.design_experiment(
    criterion            = designer.d_opt_criterion,
    solver               = "ipopt",
    solver_options       = {"linear_solver": "ma57"},
    pseudo_bayesian_type = 0,   # 0 = average FIM; 1 = average criterion
)
```

---

## V-optimal MBDoE

V-optimal design minimises model prediction variance at a specific
operating condition `dw` (e.g. the economically optimal process point),
rather than minimising global parameter uncertainty as D/A-optimal designs
do. It follows a two-stage workflow:

**Stage 1 — Process optimisation:** find `dw` by solving a constrained
nonlinear programme over the operating space via IPOPT.

**Stage 2 — V-optimal MBDoE:** design experiments that minimise
`J_V = trace(W FIM⁻¹ Wᵀ)` where `W` is the scaled sensitivity matrix
evaluated at `dw`.

```python
# Stage 1
designer.process_objective   = my_objective    # callable(tic, tvc, mp) -> float
designer.process_constraints = my_constraints  # callable(tic, tvc, mp) -> list
designer.dw_sense            = "maximize"
designer.dw_bounds_tic       = [(lb, ub), ...]

designer.find_optimal_operating_point(
    init_guess     = np.array([[60.0, 70.0, 1.0]]),
    solver         = "ipopt",
    solver_options = {"linear_solver": "ma57"},
)

# Stage 2
designer.dw_spt = np.array([t_final])
designer.design_experiment(
    criterion                = designer.v_opt_criterion,
    solver                   = "ipopt",
    solver_options           = {"linear_solver": "ma57"},
    optimize_sampling_times  = True,
)
```

See `examples/v_optimal_test_case.py` for a complete worked example with
a batch reactor system featuring three competing reactions.

> Shahmohammadi, A. & McAuley, K.B. (2019). Sequential model-based A- and
> V-optimal design of experiments for building fundamental models of
> pharmaceutical production processes. *Computers & Chemical Engineering*,
> 129, 106504. https://doi.org/10.1016/j.compchemeng.2019.06.029

---

## Examples

The `examples/` folder contains worked examples:

- `examples/v_optimal_test_case.py` — V-optimal MBDoE, batch reactor,
  three competing reactions, two-stage workflow
- `examples/first_order_reaction_ipopt.py` — D-optimal design with IPOPT,
  first-order ODE, finite-difference sensitivities
- `examples/first_order_reaction_pyomo.py` — D-optimal design with
  Pyomo.DAE and IFT sensitivities, parallel evaluation
- `examples/ivt_mbdoe.py` — industrial IVT (In-Vitro Testing) MBDoE
  application with batch reactor model

---

## Running the Tests

```bash
python pydex_full_capability_test.py
```

32 end-to-end tests covering all design criteria, both sensitivity paths
(finite-difference and Pyomo IFT), sequential vs parallel correctness,
prior FIM, save/load, and visualisation.

---

Do you have a question, suggestion, or feature request? Feel free to open
an issue or contact the original author at
[kennedy.putra.kusumo@gmail.com](mailto:kennedy.putra.kusumo@gmail.com).
