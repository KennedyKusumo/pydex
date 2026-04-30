"""
first_order_reaction_ipopt.py
==============================
D-optimal design of experiments for a first-order reaction
    A → B,   dA/dt = -k·A   →   A(t) = A₀·exp(-k·t)
We want to find the best sampling times to estimate the rate constant k
(and optionally A₀) most precisely.

Design choices
--------------
  decision variable : sampling time  t ∈ [0, 10] (the "ti_control" in pydex)
  response          : A(t)  (one measurement per sample)
  parameters        : [k, A0]   (or just [k] for the fixed-A₀ variant)

Three runs
----------
  1. Local D-optimal  — nominal k=0.5, A0=1.0  (fast, simple)
  2. Pseudo-Bayesian  — 200 prior samples of k ~ U[0.1, 1.0], A0=1 fixed
  3. Pseudo-Bayesian  — 200 prior samples of both [k, A0]
"""

import numpy as np
from pydex.core.designer import Designer

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
def simulate(ti_controls, model_parameters):
    """
    ti_controls       : [t]          — sampling time
    model_parameters  : [k, A0]      — rate constant, initial concentration
    returns           : [A(t)]
    """
    t  = ti_controls[0]
    k  = model_parameters[0]
    A0 = model_parameters[1]
    return np.array([A0 * np.exp(-k * t)])


# ─────────────────────────────────────────────────────────────────────────────
# Candidate sampling times: 0 to 10 in 51 steps
# ─────────────────────────────────────────────────────────────────────────────
t_max        = 10.0
n_candidates = 51
t_candidates = np.linspace(0.0, t_max, n_candidates).reshape(-1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def make_designer(model_parameters, verbose=0):
    d = Designer()
    d.simulate               = simulate
    d.model_parameters       = model_parameters
    d.ti_controls_candidates = t_candidates
    d.initialize(verbose=verbose)
    return d


def print_result(d, label):
    p_min   = 1e-4
    efforts = d.efforts.flatten()
    t_vals  = t_candidates.flatten()
    supports = [(t_vals[i], efforts[i])
                for i in range(len(efforts)) if efforts[i] > p_min]
    print(f"\n  {'─'*60}")
    print(f"  {label}")
    print(f"  {'─'*60}")
    print(f"  Criterion value : {d._criterion_value:.6f}")
    print(f"  Optimal sampling times ({len(supports)} support points):")
    for t, p in sorted(supports):
        print(f"    t = {t:6.3f}   effort = {p:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# RUN 1 — Local D-optimal design
#          Nominal:  k=0.5, A0=1.0
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*64)
print("  RUN 1: Local D-optimal  (k=0.5, A0=1.0, nominal)")
print("█"*64)

theta_nom = np.array([0.5, 1.0])     # [k, A0] — 1D → local design

d1 = make_designer(theta_nom)
d1.design_experiment(
    d1.d_opt_criterion,
    package="ipopt",
    optimizer="ma57",
    opt_options={"tol": 1e-10, "max_iter": 3000},
)
print_result(d1, "IPOPT / MA57 — local D-optimal")


# ═════════════════════════════════════════════════════════════════════════════
# RUN 2 — Pseudo-Bayesian D-optimal design
#          k ~ Uniform[0.1, 1.0],  A0 = 1.0 fixed
#          We embed A0 in model_parameters too so the signature is consistent.
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*64)
print("  RUN 2: Pseudo-Bayesian D-optimal  (k uncertain, A0=1 fixed)")
print("█"*64)

N_scr = 200
k_samples  = np.random.uniform(0.1, 1.0, N_scr)
scenarios2 = np.column_stack([k_samples, np.ones(N_scr)])  # shape (200, 2)

d2 = make_designer(scenarios2)
d2.design_experiment(
    d2.d_opt_criterion,
    package="ipopt",
    optimizer="ma57",
    pseudo_bayesian_type=0,   # 0 = average-information (avg FIM then log-det)
    opt_options={"tol": 1e-8, "max_iter": 5000},
)
print_result(d2, "IPOPT / MA57 — pseudo-Bayesian D-optimal (k uncertain)")


# ═════════════════════════════════════════════════════════════════════════════
# RUN 3 — Pseudo-Bayesian D-optimal design
#          Both k ~ U[0.1, 1.0]  and  A0 ~ U[0.5, 2.0] uncertain
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*64)
print("  RUN 3: Pseudo-Bayesian D-optimal  (both k and A0 uncertain)")
print("█"*64)

k_samples   = np.random.uniform(0.1, 1.0, N_scr)
A0_samples  = np.random.uniform(0.5, 2.0, N_scr)
scenarios3  = np.column_stack([k_samples, A0_samples])      # shape (200, 2)

d3 = make_designer(scenarios3)
d3.design_experiment(
    d3.d_opt_criterion,
    package="ipopt",
    optimizer="ma57",
    pseudo_bayesian_type=0,
    opt_options={"tol": 1e-8, "max_iter": 5000},
)
print_result(d3, "IPOPT / MA57 — pseudo-Bayesian D-optimal (k and A0 uncertain)")


# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print("  All runs complete.")
print(f"{'='*64}\n")
