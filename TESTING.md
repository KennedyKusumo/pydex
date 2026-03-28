# pydex — Minimal Test Suite

This document defines the minimal set of examples that collectively cover all major pydex
functionalities. Use this as the integration test checklist after any significant refactor.

---

## Summary Table

| Example | Features covered | Runnable without extra deps |
|---|---|---|
| `non_linear/exponential_model_1d.py` | D/A/E-opt, scipy, static nonlinear | Yes |
| `linear/order_1_polynomial_3_inputs.py` | D/A/E-opt, cvxpy, linear, apportionment, Bayesian PE | Yes (MCMC slow) |
| `prediction_oriented/order_1.py` | Prediction-oriented criteria (Dg/Di/Ag/Ai/Eg/Ei), scipy | Yes |
| `non_linear/exp_decay.py` | Pseudo-Bayesian, A-opt, scipy, static nonlinear | Yes |
| `cvar/cvar_exp_decay.py` | CVaR, risk-averse design, cvxpy | Yes |
| `parameter_estimation/estimate_rsm.py` | Parameter estimation (least-squares), static | Yes |
| `estimability/case_2_estimability.py` | Estimability study, dynamic ODE | Needs pyomo |
| `exact_experiments/exact_2d_factorial.py` | Exact/discrete design, BnB | Needs GUROBI |
| `ode/case_1.py` | Dynamic ODE, time-invariant controls, D-opt | Needs pyomo |
| `time_varying_controls/case_2_tvc.py` | Time-varying controls, dynamic ODE | Needs pyomo |

---

## Feature Coverage Matrix

✓ = tested by this example · — = not tested

| Example | D/A/E-opt | Pred-oriented | Pseudo-Bayesian | CVaR | scipy | cvxpy | Linear | Non-linear | Static | Dynamic/ODE | TV controls | Apportionment | Param. est. | Bayesian PE | Estimability | Exact design | Runnable now |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `non_linear/exponential_model_1d.py` | ✓ | — | — | — | ✓ | — | — | ✓ | ✓ | — | — | ✓ | — | — | — | — | ✓ |
| `linear/order_1_polynomial_3_inputs.py` | ✓ | — | — | — | — | ✓ | ✓ | — | ✓ | — | — | ✓ | — | ✓ | — | — | ✓ |
| `prediction_oriented/order_1.py` | — | ✓ | — | — | ✓ | — | ✓ | — | ✓ | — | — | — | — | — | — | — | ✓ |
| `non_linear/exp_decay.py` | ✓ | — | ✓ | — | ✓ | — | — | ✓ | ✓ | — | — | — | — | — | — | — | ✓ |
| `cvar/cvar_exp_decay.py` | — | — | — | ✓ | — | ✓ | — | ✓ | ✓ | — | — | — | — | — | — | — | ✓ |
| `parameter_estimation/estimate_rsm.py` | — | — | — | — | ✓ | — | — | ✓ | ✓ | — | — | — | ✓ | — | — | — | ✓ |
| `estimability/case_2_estimability.py` | — | — | — | — | ✓ | — | — | ✓ | — | ✓ | — | — | — | — | ✓ | — | Needs pyomo |
| `exact_experiments/exact_2d_factorial.py` | ✓ | — | — | — | — | ✓ | ✓ | — | ✓ | — | — | — | — | — | — | ✓ | Needs GUROBI |
| `ode/case_1.py` | ✓ | — | — | — | — | ✓ | — | ✓ | — | ✓ | — | ✓ | — | — | — | — | Needs pyomo |
| `time_varying_controls/case_2_tvc.py` | ✓ | — | — | — | — | ✓ | — | ✓ | — | ✓ | ✓ | — | — | — | — | — | Needs pyomo |

---

## Coverage Gaps

Features only covered by examples that need extra dependencies:

- **Dynamic/ODE systems** — only `ode/case_1.py` and `case_2_tvc.py` (both need pyomo)
- **Time-varying controls** — only `case_2_tvc.py` (needs pyomo)
- **Exact/discrete design** — only `exact_2d_factorial.py` (needs GUROBI)
- **Estimability** — only `case_2_estimability.py` (needs pyomo)

These gaps are environment constraints, not missing functionality. Once pyomo and a
MIP solver are available, the full suite can be run.

---

## How to Run

```bash
# Run all currently-runnable examples
cd examples
MPLBACKEND=Agg python non_linear/exponential_model_1d.py
MPLBACKEND=Agg python linear/order_1_polynomial_3_inputs.py
MPLBACKEND=Agg python prediction_oriented/order_1.py
MPLBACKEND=Agg python non_linear/exp_decay.py
MPLBACKEND=Agg python cvar/cvar_exp_decay.py
MPLBACKEND=Agg python parameter_estimation/estimate_rsm.py
```
