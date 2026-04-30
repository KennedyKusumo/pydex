"""
v_optimal_test_case.py
======================
Example of V-optimal Model-Based Design of Experiments (MBDoE) using pydex.

Overview
--------
This example demonstrates a two-stage workflow for designing experiments that
are specifically optimised for prediction accuracy at a target operating
condition, rather than globally minimising parameter uncertainty.

    Stage 1 — Process optimisation:
        Find the operating point dw = (T0, Tjacket, catalyst_load) that
        maximises product yield subject to quality and safety constraints.
        This is solved as a nonlinear programme via IPOPT.

    Stage 2 — V-optimal MBDoE:
        Design experiments that minimise model prediction variance at dw,
        using the V-optimality criterion:

            J_V = trace( W * FIM^{-1} * W^T )

        where W is the scaled sensitivity matrix evaluated at dw.
        The resulting design is compared against A-optimal and D-optimal
        designs to quantify the benefit of prediction-oriented design.

Why V-optimal?
--------------
Classical D-optimal and A-optimal designs minimise parameter uncertainty
globally across the entire operating space. When the ultimate goal is to
deploy the model at a specific operating condition (e.g. the economically
optimal process point), these global designs may allocate experimental
effort to regions that are informative about parameters in general but not
about predictions at the point of interest.

V-optimal design targets the FIM inversion specifically towards the
prediction directions encoded in W, concentrating experimental effort
near dw. The trade-off is that V-optimal designs are less informative
away from dw — which is acceptable when the operating point is known.

Reference:
    Shahmohammadi, A. & McAuley, K.B. (2019). Sequential model-based A- and
    V-optimal design of experiments for building fundamental models of
    pharmaceutical production processes. Computers & Chemical Engineering,
    129, 106504. https://doi.org/10.1016/j.compchemeng.2019.06.029

Reaction system
---------------
Three parallel first-order reactions consuming reactant A:

    A -> B    desired product    endothermic   Ea_main = 55,000 J/mol
    A -> I    impurity           exothermic    Ea_imp  = 75,000 J/mol
    A -> D    decomposition      exothermic    Ea_dec  = 80,000 J/mol
                                              (reference temp = 85 C)

Because Ea_imp and Ea_dec are both larger than Ea_main, high temperatures
accelerate the side reactions disproportionately. This creates a genuine
three-way trade-off between productivity and quality.

The energy balance includes jacket heat transfer (U = 5000 J/hr/K), making
Tjacket a meaningful degree of freedom — the reactor temperature tracks
the jacket temperature rather than staying at T0.

Process optimisation constraints
---------------------------------
    CI_final <= 0.05 mol/L    impurity quality specification
    CD_final <= 0.05 mol/L    decomposition byproduct specification
    Tjacket  >= T0            physical (jacket must heat, not cool)

The optimal operating point sits in the interior of the feasible region,
where both quality constraints are simultaneously active.

Parameters estimated
--------------------
    theta = [k_ref, Ea, k_ref_imp, Ea_imp, k_ref_dec, Ea_dec]

    k_ref     : main reaction rate constant at T_ref = 60 C  (1/hr)
    Ea        : main reaction activation energy               (J/mol)
    k_ref_imp : impurity rate constant at T_ref = 60 C       (1/hr)
    Ea_imp    : impurity activation energy                    (J/mol)
    k_ref_dec : decomposition rate constant at T_ref = 85 C  (1/hr)
    Ea_dec    : decomposition activation energy               (J/mol)

Pydex interface
---------------
Signature type 2: simulate(ti_controls, sampling_times, model_parameters)

    ti_controls      : [T0 (C), Tjacket (C), catalyst_load]
    sampling_times   : 1-D array of measurement times (hr)
    model_parameters : [k_ref, Ea, k_ref_imp, Ea_imp, k_ref_dec, Ea_dec]
    returns          : np.ndarray shape (n_spt, 4) — [CA, CB, CI, CD]

Dependencies
------------
    numpy, scipy, matplotlib
    pydex >= 0.0.9 with V-optimal extensions:
        Designer.find_optimal_operating_point()
        Designer.design_v_optimal()
        Designer.v_opt_criterion()
        Designer._eval_W_matrix()

Solver
------
    IPOPT is used for both Stage 1 and Stage 2. The default linear solver
    is "ma57" (HSL). If HSL is not available, change optimizer="ma57" to
    optimizer="mumps" in all three design calls and in
    find_optimal_operating_point().

Usage
-----
    Run directly:   python v_optimal_test_case.py
    In Spyder:      F5  (or Run > Run File)

    The script produces four sets of figures:
      1. Representative model trajectories at three operating conditions
      2. Constraint landscape (CB, CI, CD vs T0 and catalyst load)
      3. Trajectory at the optimal operating point
      4. Design comparison: V-optimal vs A-optimal vs D-optimal
         - Bubble chart of effort allocation in (T0, Tjacket) space
         - Concentration trajectories at selected candidates
         - Bar chart of J_V for each design
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pydex.core.designer import Designer


# =============================================================================
# CONFIG — edit here to explore different scenarios
# =============================================================================

# IPOPT linear solver.  Use "mumps" if HSL (ma57) is not available.
LINEAR_SOLVER = "ma57"

# Measurement noise standard deviation for each response (mol/L).
# Used to build the diagonal error covariance matrix passed to pydex.
SIGMA_RESPONSES = 0.01   # std dev ~1% of CA0 for all four species

# Quality and safety constraints (mol/L at t = T_FINAL)
CI_MAX = 0.05   # maximum allowable impurity concentration
CD_MAX = 0.05   # maximum allowable decomposition product concentration


# =============================================================================
# Fixed physical parameters
# (not estimated — treated as known from independent experiments)
# =============================================================================
R           = 8.314       # universal gas constant              (J/mol/K)
T_ref_C     = 60.0        # Arrhenius reference temp, A->B, A->I (C)
T_ref_K     = T_ref_C + 273.15
T_ref_dec_C = 85.0        # Arrhenius reference temp, A->D        (C)
T_ref_dec_K = T_ref_dec_C + 273.15
Hr_main     =  50000.0    # heat of reaction A->B  (J/mol, +ve = endothermic)
Hr_imp      = -30000.0    # heat of reaction A->I  (J/mol, -ve = exothermic)
Hr_dec      = -60000.0    # heat of reaction A->D  (J/mol, stronger exotherm)
Cp          = 4184.0      # heat capacity of reaction mixture    (J/kg/K)
mass        = 1.0         # reactor contents mass                (kg)
U           = 5000.0      # overall heat transfer coefficient    (J/hr/K)
A_area      = 1.0         # heat transfer area                   (m^2)
CA0_fixed   = 1.0         # initial concentration of A           (mol/L, fixed)
T_FINAL     = 1.0         # batch end time                       (hr)


# =============================================================================
# Parameter values
# =============================================================================

# True values — used to generate in-silico data and for plotting
THETA_TRUE = np.array([
    1.0,       # k_ref      (1/hr)   main reaction rate at T_ref
    55000.0,   # Ea         (J/mol)  main reaction activation energy
    0.08,      # k_ref_imp  (1/hr)   impurity rate at T_ref
    75000.0,   # Ea_imp     (J/mol)  impurity activation energy (> Ea)
    0.3,       # k_ref_dec  (1/hr)   decomposition rate at T_ref_dec
    80000.0,   # Ea_dec     (J/mol)  decomposition activation energy (> Ea_imp)
])

# Initial guess — represents knowledge available before experiments
# Purposely offset from THETA_TRUE to simulate realistic prior uncertainty
THETA_GUESS = np.array([
    0.8,       # k_ref      — 20% low
    50000.0,   # Ea         — ~9% low
    0.06,      # k_ref_imp  — 25% low
    70000.0,   # Ea_imp     — ~7% low
    0.2,       # k_ref_dec  — 33% low
    75000.0,   # Ea_dec     — ~6% low
])

PARAM_NAMES = ["k_ref", "Ea", "k_ref_imp", "Ea_imp", "k_ref_dec", "Ea_dec"]


# =============================================================================
# ODE system
# =============================================================================

def _odes(t, y, Tjacket_K, cat, k_ref, Ea, k_ref_imp, Ea_imp, k_ref_dec, Ea_dec):
    """
    ODE right-hand side for the three-reaction batch system.

    State vector y = [CA, CB, CI, CD, T]
        CA : concentration of reactant A         (mol/L)
        CB : concentration of desired product B  (mol/L)
        CI : concentration of impurity I         (mol/L)
        CD : concentration of decomposition D    (mol/L)
        T  : reactor temperature                 (K)

    All three reactions are first-order in CA and follow a modified
    Arrhenius expression with a reference-temperature formulation:

        k(T) = k_ref * catalyst_load * exp( -Ea/R * (1/T - 1/T_ref) )

    This form avoids the numerical sensitivity of the standard Arrhenius
    pre-exponential factor A = k_ref * exp(Ea / (R * T_ref)) when Ea is large.

    Reactions:
        A -> B   r_main = k_main(T) * CA   endothermic, T_ref = 60 C
        A -> I   r_imp  = k_imp(T)  * CA   exothermic,  T_ref = 60 C
        A -> D   r_dec  = k_dec(T)  * CA   exothermic,  T_ref = 85 C

    The A->D reaction uses a higher reference temperature (85 C) reflecting
    the fact that it only becomes significant at elevated temperatures.

    Energy balance:
        mass * Cp * dT/dt = Q - Hr_main*r_main - Hr_imp*r_imp - Hr_dec*r_dec
        Q = U * A_area * (Tjacket_K - T)

    Note on sign convention: Hr > 0 means endothermic (absorbs heat, cools
    reactor). Hr < 0 means exothermic (releases heat, warms reactor).
    The energy balance therefore subtracts Hr*r, so an endothermic reaction
    (positive Hr) reduces dT/dt and an exothermic one (negative Hr) increases it.

    Parameters
    ----------
    t          : float        current time (hr)
    y          : list[float]  state vector [CA, CB, CI, CD, T(K)]
    Tjacket_K  : float        jacket temperature (K)
    cat        : float        catalyst load (dimensionless multiplier on k_ref)
    k_ref      : float        main reaction rate constant at T_ref (1/hr)
    Ea         : float        main reaction activation energy (J/mol)
    k_ref_imp  : float        impurity rate constant at T_ref (1/hr)
    Ea_imp     : float        impurity activation energy (J/mol)
    k_ref_dec  : float        decomposition rate constant at T_ref_dec (1/hr)
    Ea_dec     : float        decomposition activation energy (J/mol)

    Returns
    -------
    list[float] : [dCA/dt, dCB/dt, dCI/dt, dCD/dt, dT/dt]
    """
    CA, CB, CI, CD, T = y
    CA = max(CA, 0.0)   # guard against numerical undershoot below zero

    k_main = max(k_ref     * cat * np.exp(-Ea     / R * (1/T - 1/T_ref_K)),     0.0)
    k_imp  = max(k_ref_imp * cat * np.exp(-Ea_imp / R * (1/T - 1/T_ref_K)),     0.0)
    k_dec  = max(k_ref_dec * cat * np.exp(-Ea_dec / R * (1/T - 1/T_ref_dec_K)), 0.0)

    r_main = k_main * CA   # rate of A->B  (mol/L/hr)
    r_imp  = k_imp  * CA   # rate of A->I  (mol/L/hr)
    r_dec  = k_dec  * CA   # rate of A->D  (mol/L/hr)

    # Mass balances: A is consumed by all three pathways
    # B, I, D each produced by their respective pathway only
    dCA = -(r_main + r_imp + r_dec)
    dCB =   r_main
    dCI =   r_imp
    dCD =   r_dec

    # Energy balance: jacket supplies heat Q, reactions absorb/release heat
    Q  = U * A_area * (Tjacket_K - T)
    dT = (Q - Hr_main*r_main - Hr_imp*r_imp - Hr_dec*r_dec) / (mass * Cp)

    return [dCA, dCB, dCI, dCD, dT]


def _solve(T0, Tjacket, catalyst_load, mp, t_eval):
    """
    Integrate the ODE system from t=0 to t=t_eval[-1].

    Uses Radau (implicit Runge-Kutta, order 5), which is well-suited for
    the moderately stiff dynamics that arise at high temperatures when the
    decomposition reaction activates sharply.

    Parameters
    ----------
    T0             : float        initial reactor temperature (C)
    Tjacket        : float        jacket temperature (C)
    catalyst_load  : float        catalyst load (dimensionless)
    mp             : array-like   model parameters [k_ref, Ea, k_ref_imp,
                                  Ea_imp, k_ref_dec, Ea_dec]
    t_eval         : np.ndarray   time points at which to store solution (hr).
                                  When only the endpoint is needed (e.g. in
                                  the process optimisation), pass
                                  np.array([T_FINAL]) to avoid storing the
                                  full trajectory.

    Returns
    -------
    scipy OdeResult
        .y  : shape (5, len(t_eval)) — rows are [CA, CB, CI, CD, T(K)]
        .t  : shape (len(t_eval),)   — time points (hr)
    """
    sol = solve_ivp(
        _odes,
        (0.0, t_eval[-1]),
        [CA0_fixed, 0.0, 0.0, 0.0, T0 + 273.15],   # initial conditions
        args=(Tjacket + 273.15, catalyst_load, *mp),
        t_eval=t_eval,
        method='Radau',
        rtol=1e-8,
        atol=1e-10,
    )
    return sol


# =============================================================================
# Pydex simulate function  (signature type 2)
# =============================================================================

def simulate(ti_controls, sampling_times, model_parameters):
    """
    Pydex-compatible simulate function.

    This function is assigned to designer.simulate and is called internally
    by pydex during sensitivity analysis and optimal design.  It must follow
    pydex signature type 2: simulate(ti_controls, sampling_times, model_parameters).

    Parameters
    ----------
    ti_controls      : array-like, length 3
                       [T0 (C), Tjacket (C), catalyst_load]
    sampling_times   : 1-D array of measurement times (hr)
                       Pydex passes the candidate sampling times here.
                       When optimize_sampling_times=True, pydex optimises
                       over these times as additional decision variables.
    model_parameters : 1-D array, length 6
                       [k_ref, Ea, k_ref_imp, Ea_imp, k_ref_dec, Ea_dec]
                       During sensitivity analysis, pydex perturbs these
                       values via finite differences to build the FIM.

    Returns
    -------
    np.ndarray, shape (n_spt, 4)
        Columns: [CA, CB, CI, CD] — concentrations at each sampling time (mol/L).
        CA0 is fixed at CA0_fixed = 1.0 mol/L for all experiments.
        Temperature (state index 4) is not returned as it is not directly
        measured, but it drives the ODE dynamics through the energy balance.
    """
    T0, Tjacket, cat = ti_controls
    sol = _solve(T0, Tjacket, cat, model_parameters, sampling_times)
    return np.column_stack([sol.y[0], sol.y[1], sol.y[2], sol.y[3]])


# =============================================================================
# Stage 1: process objective and constraints
# =============================================================================

def process_objective(tic, tvc, mp):
    """
    Process optimisation objective for Stage 1.

    Maximises CB at the end of the batch (t = T_FINAL).  Since CA0 is
    fixed, maximising CB is equivalent to maximising the yield of B.

    The quality constraints (CI_MAX, CD_MAX) prevent the optimizer from
    achieving high CB via aggressive high-temperature routes that produce
    excessive impurity or decomposition product.  The result is a solution
    that sits on the constraint boundary — the best achievable yield given
    the product quality requirements.

    Parameters
    ----------
    tic : array-like  [T0 (C), Tjacket (C), catalyst_load]
    tvc : array-like  time-varying controls (unused — empty list)
    mp  : array-like  current model parameters

    Returns
    -------
    float : CB at t = T_FINAL (mol/L)  — to be maximised
    """
    sol = _solve(tic[0], tic[1], tic[2], mp, np.array([T_FINAL]))
    return float(sol.y[1, 0])


def process_constraints(tic, tvc, mp):
    """
    Process optimisation constraints for Stage 1.

    Returns a list of constraint dictionaries in pydex / scipy format.
    Each dict has keys "type" ("ineq" or "eq") and "fun" (callable).
    For "ineq" constraints, IPOPT requires fun(tic, tvc, mp) >= 0.

    Constraints
    -----------
    1. CI_final <= CI_MAX  :  impurity quality specification
       fun = CI_MAX - CI_final >= 0

    2. CD_final <= CD_MAX  :  decomposition byproduct specification
       fun = CD_MAX - CD_final >= 0

    3. Tjacket >= T0       :  physical constraint — jacket must heat the
       fun = Tjacket - T0 >= 0   reactor, not cool it (no refrigeration)

    The CI and CD constraints create two separate feasibility boundaries
    in the (T0, Tjacket, catalyst_load) space, and the optimizer must
    navigate to the point of maximum CB that satisfies both simultaneously.

    Parameters
    ----------
    tic : array-like  [T0 (C), Tjacket (C), catalyst_load]
    tvc : array-like  time-varying controls (unused)
    mp  : array-like  current model parameters

    Returns
    -------
    list of dicts : [{"type": "ineq", "fun": callable}, ...]
    """
    def ci_con(tic, tvc, mp):
        """CI_MAX - CI_final >= 0"""
        sol = _solve(tic[0], tic[1], tic[2], mp, np.array([T_FINAL]))
        return CI_MAX - float(sol.y[2, 0])

    def cd_con(tic, tvc, mp):
        """CD_MAX - CD_final >= 0"""
        sol = _solve(tic[0], tic[1], tic[2], mp, np.array([T_FINAL]))
        return CD_MAX - float(sol.y[3, 0])

    def jacket_con(tic, tvc, mp):
        """Tjacket - T0 >= 0"""
        return tic[1] - tic[0]

    return [
        {"type": "ineq", "fun": ci_con},
        {"type": "ineq", "fun": cd_con},
        {"type": "ineq", "fun": jacket_con},
    ]


# =============================================================================
# Experimental candidate grid
# =============================================================================

# Time-invariant control candidates: [T0 (C), Tjacket (C), catalyst_load]
# Only physically valid combinations are included (Tjacket >= T0).
T0_cands      = np.array([45, 50, 55, 60, 65, 70])       # initial temperature (C)
Tjacket_cands = np.array([50, 55, 60, 65, 70, 75, 80])   # jacket temperature  (C)
cat_cands     = np.array([0.5, 0.75, 1.0, 1.25, 1.5])    # catalyst load       (-)

tic_candidates = np.array([
    [T0, Tj, cat]
    for T0  in T0_cands
    for Tj  in Tjacket_cands  if Tj >= T0   # physical constraint
    for cat in cat_cands
])   # shape (n_c, 3)

# Sampling time candidates: 20 uniformly spaced points over [0.05, 1.0] hr.
# When optimize_sampling_times=True, pydex allocates effort over these
# times as additional decision variables alongside the candidate selection.
spt_grid       = np.linspace(0.05, 1.0, 20)                          # (hr)
spt_candidates = np.tile(spt_grid, (len(tic_candidates), 1))          # (n_c, 20)


# =============================================================================
# Plotting helpers
# =============================================================================

def plot_case(T0, Tjacket, catalyst_load, title, mp=THETA_TRUE):
    """
    Plot concentration trajectories, temperature profile, selectivity,
    and yield breakdown for a single operating condition.

    Four panels:
      1. Concentrations of CA, CB, CI, CD vs time with constraint lines
      2. Reactor temperature vs time
      3. Instantaneous selectivity to B (CB / (CB + CI + CD)) vs time
      4. Yield of each species as % of CA0 vs time

    Parameters
    ----------
    T0, Tjacket, catalyst_load : float  operating conditions
    title : str                         figure title
    mp    : array-like                  model parameters (default: THETA_TRUE)
    """
    t_eval = np.linspace(0.0, T_FINAL, 300)
    sol    = _solve(T0, Tjacket, catalyst_load, mp, t_eval)
    CA = sol.y[0]; CB = sol.y[1]; CI = sol.y[2]
    CD = sol.y[3]; T  = sol.y[4] - 273.15

    converted = CB + CI + CD
    SB = np.where(converted > 1e-6, CB / converted * 100, 100.0)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(title, fontsize=11)

    axes[0].plot(t_eval, CA, 'b', label='CA')
    axes[0].plot(t_eval, CB, 'g', label='CB')
    axes[0].plot(t_eval, CI, 'r', label='CI')
    axes[0].plot(t_eval, CD, 'm', label='CD')
    axes[0].axhline(CI_MAX, color='r', ls='--', alpha=0.5,
                    label=f'CI_max={CI_MAX}')
    axes[0].axhline(CD_MAX, color='m', ls='--', alpha=0.5,
                    label=f'CD_max={CD_MAX}')
    axes[0].set(xlabel='time (hr)', ylabel='Concentration (mol/L)')
    axes[0].legend(fontsize=7)

    axes[1].plot(t_eval, T, 'k')
    axes[1].set(xlabel='time (hr)', ylabel='T (C)')

    axes[2].plot(t_eval, SB, 'g')
    axes[2].set(xlabel='time (hr)', ylabel='Selectivity to B (%)',
                ylim=[0, 105])

    axes[3].plot(t_eval, CB / CA0_fixed * 100, 'g', label='B yield')
    axes[3].plot(t_eval, CI / CA0_fixed * 100, 'r', label='I yield')
    axes[3].plot(t_eval, CD / CA0_fixed * 100, 'm', label='D yield')
    axes[3].set(xlabel='time (hr)', ylabel='Yield (% of CA0)')
    axes[3].legend(fontsize=7)

    plt.tight_layout()
    return fig


def plot_constraint_landscape(dw_tic, mp=THETA_GUESS):
    """
    Visualise the process optimisation landscape.

    Produces a 3-row contour map showing CB, CI, and CD at t=T_FINAL
    as functions of (T0, catalyst_load) for several fixed Tjacket values.
    A column is always included for the optimal Tjacket from dw_tic.

    Constraint boundaries are overlaid as dashed lines:
      - Red dashed  : CI = CI_MAX
      - Magenta dashed : CD = CD_MAX

    The optimal operating point dw is marked with a gold star in its
    corresponding Tjacket column.

    All columns share the same colorbar scale, enabling direct comparison
    of how CB, CI, and CD change with Tjacket.

    Parameters
    ----------
    dw_tic : np.ndarray, shape (r_w, 3)
             Optimal operating points from find_optimal_operating_point().
             Multiple rows arise when multiple starting points are used;
             duplicates are removed automatically.
    mp     : array-like   model parameters for evaluation (default: THETA_GUESS)
    """
    T0_grid  = np.linspace(45, 72, 40)
    cat_grid = np.linspace(0.5, 2.0, 40)
    T0_mg, cat_mg = np.meshgrid(T0_grid, cat_grid)
    t_eval = np.array([T_FINAL])

    # remove duplicate operating points arising from multiple restarts
    _, uid    = np.unique(np.round(dw_tic, 2), axis=0, return_index=True)
    dw_unique = dw_tic[uid]

    # always include the optimal Tjacket as a column
    Tjacket_vals = sorted(set([60.0, 70.0, 80.0,
                                round(float(dw_unique[0, 1]), 1)]))
    n_cols = len(Tjacket_vals)

    # pre-compute all maps so colorbars can be shared across columns
    all_maps = []
    for Tj in Tjacket_vals:
        CB_map = np.full_like(T0_mg, np.nan)
        CI_map = np.full_like(T0_mg, np.nan)
        CD_map = np.full_like(T0_mg, np.nan)
        for i in range(T0_mg.shape[0]):
            for j in range(T0_mg.shape[1]):
                T0  = T0_mg[i, j]
                cat = cat_mg[i, j]
                if Tj < T0:
                    continue   # physically invalid
                sol = _solve(T0, Tj, cat, mp, t_eval)
                CB_map[i, j] = sol.y[1, 0]
                CI_map[i, j] = sol.y[2, 0]
                CD_map[i, j] = sol.y[3, 0]
        all_maps.append((CB_map, CI_map, CD_map))

    # global ranges for shared colorbars
    CB_vmin = np.nanmin([m[0] for m in all_maps])
    CB_vmax = np.nanmax([m[0] for m in all_maps])
    CI_vmin = np.nanmin([m[1] for m in all_maps])
    CI_vmax = np.nanmax([m[1] for m in all_maps])
    CD_vmin = np.nanmin([m[2] for m in all_maps])
    CD_vmax = np.nanmax([m[2] for m in all_maps])
    CB_levels = np.linspace(CB_vmin, CB_vmax, 16)
    CI_levels = np.linspace(CI_vmin, CI_vmax, 16)
    CD_levels = np.linspace(CD_vmin, CD_vmax, 16)

    fig, axes = plt.subplots(3, n_cols, figsize=(5 * n_cols, 12),
                             sharex=True, sharey=True)
    if n_cols == 1:
        axes = axes.reshape(3, 1)

    fig.suptitle(
        f'Constraint landscape at t={T_FINAL} hr  |  '
        f'Red = CI_MAX={CI_MAX}  Magenta = CD_MAX={CD_MAX}  '
        f'Star = optimal dw',
        fontsize=11
    )

    for col, (Tj, (CB_map, CI_map, CD_map)) in enumerate(
            zip(Tjacket_vals, all_maps)):

        is_dw_col = abs(Tj - float(dw_unique[0, 1])) < 0.6
        col_label = f'Tjacket={Tj:.0f}C' + (' <- dw' if is_dw_col else '')

        def add_star(ax, _is_dw=is_dw_col, _dw=dw_unique):
            """Overlay gold star at optimal operating point."""
            if _is_dw:
                for w, dw_pt in enumerate(_dw):
                    ax.plot(dw_pt[0], dw_pt[2], '*', color='gold',
                            markersize=16, markeredgecolor='k',
                            markeredgewidth=0.8, zorder=5,
                            label='dw' if w == 0 else '_nolegend_')
                ax.legend(fontsize=7)

        # Row 0: CB (product)
        ax = axes[0, col]
        cf = ax.contourf(T0_mg, cat_mg, CB_map, levels=CB_levels,
                         cmap='Greens', vmin=CB_vmin, vmax=CB_vmax)
        ax.contour(T0_mg, cat_mg, CI_map, levels=[CI_MAX],
                   colors='red', linewidths=1.5, linestyles='--')
        ax.contour(T0_mg, cat_mg, CD_map, levels=[CD_MAX],
                   colors='magenta', linewidths=1.5, linestyles='--')
        ax.set_title(f'CB  |  {col_label}')
        if col == 0:
            ax.set_ylabel('Catalyst load')
        if col == n_cols - 1:
            plt.colorbar(cf, ax=axes[0, :], label='CB (mol/L)',
                         fraction=0.02, pad=0.04)
        add_star(ax)

        # Row 1: CI (impurity)
        ax2 = axes[1, col]
        cf2 = ax2.contourf(T0_mg, cat_mg, CI_map, levels=CI_levels,
                           cmap='Reds', vmin=CI_vmin, vmax=CI_vmax)
        ax2.contour(T0_mg, cat_mg, CI_map, levels=[CI_MAX],
                    colors='red', linewidths=1.5, linestyles='--')
        ax2.set_title(f'CI  |  {col_label}')
        if col == 0:
            ax2.set_ylabel('Catalyst load')
        if col == n_cols - 1:
            plt.colorbar(cf2, ax=axes[1, :], label='CI (mol/L)',
                         fraction=0.02, pad=0.04)
        add_star(ax2)

        # Row 2: CD (decomposition product)
        ax3 = axes[2, col]
        cf3 = ax3.contourf(T0_mg, cat_mg, CD_map, levels=CD_levels,
                           cmap='Purples', vmin=CD_vmin, vmax=CD_vmax)
        ax3.contour(T0_mg, cat_mg, CD_map, levels=[CD_MAX],
                    colors='magenta', linewidths=1.5, linestyles='--')
        ax3.set_title(f'CD  |  {col_label}')
        ax3.set_xlabel('T0 (C)')
        if col == 0:
            ax3.set_ylabel('Catalyst load')
        if col == n_cols - 1:
            plt.colorbar(cf3, ax=axes[2, :], label='CD (mol/L)',
                         fraction=0.02, pad=0.04)
        add_star(ax3)

    plt.tight_layout()
    return fig


def plot_design_comparison(efforts_dict, tic_candidates, dw_tic,
                           jv_dict, mp=THETA_GUESS):
    """
    Compare V-optimal, A-optimal, and D-optimal experimental designs.

    For each design, two panels are shown side by side:

    Left — Bubble chart in (T0, Tjacket) space:
        Each bubble represents a selected candidate experiment.
        Bubble size is proportional to the effort weight allocated to that
        candidate. Bubble colour encodes catalyst load.
        All candidate experiments (selected and unselected) are shown as
        faint background markers to give context.
        The gold star marks the optimal operating point dw from Stage 1.

    Right — Concentration trajectories:
        Model trajectories (CB solid, CI dashed, CD dotted) are plotted for
        each selected candidate, with opacity proportional to effort weight.
        The bold black/red/magenta lines are the trajectories at dw itself.
        Circular markers show the selected sampling times for each candidate.

    All rows share identical axis limits, enabling direct visual comparison
    of how the three designs differ in their coverage of the operating space.

    Parameters
    ----------
    efforts_dict  : dict  {name: effort_array}  — one entry per design
    tic_candidates: np.ndarray  shape (n_c, 3)  — full candidate grid
    dw_tic        : np.ndarray  shape (r_w, 3)  — optimal operating point(s)
    jv_dict       : dict  {name: J_V value}     — prediction variance per design
    mp            : array-like  model parameters for trajectory plotting
    """
    designs = list(efforts_dict.keys())
    colours = {
        'V-optimal': '#1f77b4',
        'A-optimal': '#ff7f0e',
        'D-optimal': '#2ca02c',
    }
    tol    = 1e-3
    t_plot = np.linspace(0.0, T_FINAL, 200)

    # remove duplicate operating points
    _, uid    = np.unique(np.round(dw_tic, 2), axis=0, return_index=True)
    dw_unique = dw_tic[uid]

    # axis limits that include the dw point
    T0_all   = np.append(tic_candidates[:, 0], dw_unique[:, 0])
    Tj_all   = np.append(tic_candidates[:, 1], dw_unique[:, 1])
    bub_xlim = (T0_all.min() - 2, T0_all.max() + 2)
    bub_ylim = (Tj_all.min() - 2, Tj_all.max() + 4)

    # global trajectory y-limits (shared across all rows)
    all_CB, all_CI, all_CD = [], [], []
    for name, eff in efforts_dict.items():
        n_c    = len(tic_candidates)
        eff_2d = np.array(eff).reshape(n_c, -1)
        eff_pc = eff_2d.sum(axis=1)
        for idx in np.where(eff_pc > tol)[0]:
            sol = _solve(*tic_candidates[idx], mp, t_plot)
            all_CB.append(sol.y[1])
            all_CI.append(sol.y[2])
            all_CD.append(sol.y[3])
    for dw_pt in dw_unique:
        sol_dw = _solve(*dw_pt, mp, t_plot)
        all_CB.append(sol_dw.y[1])
        all_CI.append(sol_dw.y[2])
        all_CD.append(sol_dw.y[3])
    traj_ymax = max(np.concatenate(all_CB).max(),
                    np.concatenate(all_CI).max(),
                    np.concatenate(all_CD).max()) * 1.15
    traj_ylim = (-0.02, traj_ymax)

    fig, axes = plt.subplots(len(designs), 2,
                             figsize=(14, 4.5 * len(designs)))
    fig.suptitle(
        'Design comparison  |  Bubble size = effort weight  |  '
        'Star = optimal operating point dw',
        fontsize=11
    )

    for row, name in enumerate(designs):
        ax_bub  = axes[row, 0]
        ax_traj = axes[row, 1]

        n_c     = len(tic_candidates)
        eff_2d  = np.array(efforts_dict[name]).reshape(n_c, -1)
        eff_pc  = eff_2d.sum(axis=1)   # total effort per candidate
        sup_idx = np.where(eff_pc > tol)[0]
        T0s     = tic_candidates[sup_idx, 0]
        Tjs     = tic_candidates[sup_idx, 1]
        cats    = tic_candidates[sup_idx, 2]
        effs    = eff_pc[sup_idx]

        # background: all candidates as faint markers
        ax_bub.scatter(
            tic_candidates[:, 0], tic_candidates[:, 1],
            s=60, c=tic_candidates[:, 2], cmap='plasma',
            alpha=0.12, edgecolors='grey', linewidths=0.3,
            vmin=cat_cands.min(), vmax=cat_cands.max(), zorder=1,
        )
        # foreground: selected candidates sized by effort
        sc = ax_bub.scatter(
            T0s, Tjs,
            s=effs / effs.max() * 800,
            c=cats, cmap='plasma', alpha=0.85,
            edgecolors='k', linewidths=0.8,
            vmin=cat_cands.min(), vmax=cat_cands.max(), zorder=3,
        )
        plt.colorbar(sc, ax=ax_bub, label='Catalyst load')

        for w, dw_pt in enumerate(dw_unique):
            ax_bub.plot(dw_pt[0], dw_pt[1], '*', color='gold',
                        markersize=16, markeredgecolor='k', zorder=5,
                        label='dw' if w == 0 else '_nolegend_')

        ax_bub.set_xlabel('T0 (C)')
        ax_bub.set_ylabel('Tjacket (C)')
        ax_bub.set_title(
            f'{name}   J_V = {jv_dict[name]:.4f}  '
            f'({len(sup_idx)} support point(s))'
        )
        ax_bub.set_xlim(bub_xlim)
        ax_bub.set_ylim(bub_ylim)
        ax_bub.legend(fontsize=7, loc='upper left')

        # trajectories at selected candidates
        n_spt = eff_2d.shape[1]
        for idx in sup_idx:
            T0c, Tjc, catc = tic_candidates[idx]
            sol   = _solve(T0c, Tjc, catc, mp, t_plot)
            alpha = float(eff_pc[idx]) / float(eff_pc[sup_idx].max())
            ax_traj.plot(t_plot, sol.y[1], color=colours[name],
                         alpha=max(0.2, alpha), linewidth=1.2)
            ax_traj.plot(t_plot, sol.y[2], 'r--',
                         alpha=max(0.15, alpha * 0.6), linewidth=1.0)
            ax_traj.plot(t_plot, sol.y[3], 'm:',
                         alpha=max(0.15, alpha * 0.6), linewidth=1.0)
            # selected sampling times as scatter markers
            spt_eff  = eff_2d[idx, :n_spt]
            spt_here = spt_grid[:n_spt]
            sel_spt  = spt_here[spt_eff > tol]
            if len(sel_spt) > 0:
                sol_s = _solve(T0c, Tjc, catc, mp, sel_spt)
                ax_traj.scatter(
                    sel_spt, sol_s.y[1],
                    color=colours[name], s=40, zorder=6,
                    alpha=max(0.5, alpha),
                    edgecolors='k', linewidths=0.4,
                )

        # dw trajectory highlighted with white halo for visibility
        for w, dw_pt in enumerate(dw_unique):
            sol_dw = _solve(*dw_pt, mp, t_plot)
            for y_idx, col, lbl in [
                (1, 'k',           'CB@dw'),
                (2, 'darkred',     'CI@dw'),
                (3, 'darkmagenta', 'CD@dw'),
            ]:
                # white halo drawn first, then coloured line on top
                ax_traj.plot(t_plot, sol_dw.y[y_idx], 'w-',
                             linewidth=5.0, zorder=4)
                ax_traj.plot(t_plot, sol_dw.y[y_idx], color=col,
                             linewidth=2.5, zorder=5,
                             label=lbl if w == 0 else '_nolegend_')

        ax_traj.axhline(CI_MAX, color='red',     ls=':', alpha=0.6)
        ax_traj.axhline(CD_MAX, color='magenta', ls=':', alpha=0.6)
        ax_traj.set_xlabel('time (hr)')
        ax_traj.set_ylabel('Concentration (mol/L)')
        ax_traj.set_ylim(traj_ylim)
        ax_traj.set_title(
            f'{name}: solid = CB  dashed = CI  dotted = CD\n'
            f'dots = selected sampling times  |  opacity proportional to effort'
        )
        ax_traj.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    return fig


def plot_jv_bar(jv_dict):
    """
    Bar chart comparing prediction variance J_V across designs.

    J_V = trace(W * FIM^{-1} * W^T) evaluated at dw for each design.
    Lower J_V means lower prediction uncertainty at the operating point
    of interest.  V-optimal minimises J_V by construction and therefore
    always achieves the lowest bar.  The ratios quantify the cost of
    using a globally-oriented criterion (A or D) when predictions at a
    specific operating point are what matters.

    Parameters
    ----------
    jv_dict : dict  {design_name: J_V_value}
    """
    names  = list(jv_dict.keys())
    values = [jv_dict[n] for n in names]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        names, values,
        color=['#1f77b4', '#ff7f0e', '#2ca02c'],
        edgecolor='k', linewidth=0.7,
    )
    ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
    ax.set_ylabel('J_V  (prediction variance at dw)')
    ax.set_title(
        'Prediction variance at the optimal operating point\n'
        'Lower is better — V-optimal minimises this by construction'
    )
    ax.set_ylim(0, max(values) * 1.2)
    plt.tight_layout()
    return fig


# =============================================================================
# Main script
# =============================================================================

if __name__ == "__main__":

    # ── 0. Explore model behaviour before running MBDoE ───────────────────────
    # Three representative cases illustrating the trade-off:
    #   Low T:  feasible but slow — both constraints satisfied with slack
    #   Mid T:  approaching the feasibility boundary
    #   High T: fast conversion but both CI and CD constraints violated
    print("Plotting representative model cases...")
    plot_case(50, 60, 1.0, 'Low T  — slow, both constraints satisfied',
              mp=THETA_TRUE)
    plot_case(58, 68, 1.0, 'Mid T  — approaching constraint boundaries',
              mp=THETA_TRUE)
    plot_case(65, 80, 1.5, 'High T — fast, both constraints violated',
              mp=THETA_TRUE)
    plt.show()

    # ── 1. Initialise the pydex designer ──────────────────────────────────────
    print("\nInitialising pydex designer...")
    designer = Designer()
    designer.simulate                  = simulate
    designer.model_parameters          = THETA_GUESS
    designer.ti_controls_candidates    = tic_candidates
    designer.sampling_times_candidates = spt_candidates
    # diagonal error covariance: sigma^2 on each of the 4 responses
    designer.error_cov = np.diag([SIGMA_RESPONSES**2] * 4)
    designer.model_parameters_names    = PARAM_NAMES
    designer.ti_controls_names         = ["T0_C", "Tjacket_C", "catalyst_load"]
    designer.response_names            = ["CA", "CB", "CI", "CD"]
    designer.initialize(verbose=1)
    print(f"  Candidate experiments: {len(tic_candidates)}")
    print(f"  Sampling time candidates per experiment: {len(spt_grid)}")
    print(f"  Model parameters: {len(THETA_GUESS)}")

    # ── 2. Stage 1: process optimisation — find dw ────────────────────────────
    # Solve the constrained process optimisation to find the operating
    # condition dw = (T0, Tjacket, catalyst_load) that maximises CB
    # subject to the quality and physical constraints.
    #
    # Multiple starting points are used because IPOPT is a local solver.
    # The problem is well-conditioned here and all starts converge to the
    # same solution, but this is good practice for general use.
    print("\n" + "=" * 60)
    print("STAGE 1: Process optimisation — finding optimal dw")
    print("=" * 60)

    designer.process_objective   = process_objective
    designer.process_constraints = process_constraints
    designer.dw_sense            = "maximize"
    designer.dw_bounds_tic       = [
        (45.0, 70.0),   # T0        (C)
        (50.0, 85.0),   # Tjacket   (C)  — capped at 85C (decomp dominates above)
        (0.5,  2.0),    # catalyst_load
    ]
    designer.dw_bounds_tvc = []   # no time-varying controls in this model

    init_guesses = np.array([
        [55.0, 65.0, 1.0],    # start 1: moderate conditions
        [50.0, 70.0, 1.5],    # start 2: lower T, higher cat
        [60.0, 75.0, 0.75],   # start 3: higher T, lower cat
    ])

    dw_tic, dw_tvc = designer.find_optimal_operating_point(
        init_guess  = init_guesses,
        optimizer   = LINEAR_SOLVER,
        n_starts    = 2,
        opt_options = {"tol": 1e-8, "max_iter": 500},
    )

    # Report results — tolerance of 1e-6 accounts for floating-point equality
    print("\nOptimal operating point(s):")
    for w in range(dw_tic.shape[0]):
        T0_opt, Tj_opt, cat_opt = dw_tic[w]
        sol  = _solve(T0_opt, Tj_opt, cat_opt, THETA_GUESS, np.array([T_FINAL]))
        CB_f = float(sol.y[1, 0])
        CI_f = float(sol.y[2, 0])
        CD_f = float(sol.y[3, 0])
        ci_status = 'OK'   if CI_f <= CI_MAX + 1e-6 else 'VIOLATED'
        cd_status = 'OK'   if CD_f <= CD_MAX + 1e-6 else 'VIOLATED'
        print(f"  [{w+1}]  T0={T0_opt:.2f} C  Tjacket={Tj_opt:.2f} C  "
              f"catalyst={cat_opt:.3f}")
        print(f"         CB={CB_f:.4f}  CI={CI_f:.4f} ({ci_status})"
              f"  CD={CD_f:.4f} ({cd_status})")

    # Deduplicate — multiple starts may converge to the same point
    _, uid = np.unique(np.round(dw_tic, 3), axis=0, return_index=True)
    dw_tic = dw_tic[uid]
    designer.dw_tic = dw_tic   # update designer with deduplicated result
    print(f"\n  {len(dw_tic)} unique operating point(s) retained.")

    # Visualise the constraint landscape and the optimal point
    print("\nPlotting constraint landscape...")
    plot_constraint_landscape(dw_tic, mp=THETA_GUESS)
    best = dw_tic[0]
    plot_case(
        best[0], best[1], best[2],
        f"Optimal dw: T0={best[0]:.1f} C  Tjacket={best[1]:.1f} C  "
        f"catalyst={best[2]:.2f}  (on constraint boundary)",
        mp=THETA_GUESS,
    )
    plt.show()

    # ── 3. Stage 2: V-optimal design ──────────────────────────────────────────
    # Design experiments that minimise prediction variance at dw.
    # dw_spt specifies the time point(s) at which prediction accuracy matters.
    # Here we care about predictions at the end of the batch (t = T_FINAL).
    #
    # The W matrix is built internally by evaluating model sensitivities at
    # dw_tic / dw_tvc / dw_spt. The FIM is built from the candidate grid as
    # usual. The criterion J_V = trace(W * FIM^{-1} * W^T) is minimised.
    print("\n" + "=" * 60)
    print("STAGE 2: V-optimal experimental design")
    print("=" * 60)

    designer.dw_spt = np.array([T_FINAL])   # care about predictions at t=1 hr

    designer.design_v_optimal(
        package               = "ipopt",
        optimizer             = LINEAR_SOLVER,
        opt_options           = {"tol": 1e-8, "max_iter": 1000},
        regularize_fim        = False,
        optimize_sampling_times = True,   # let pydex choose when to sample
    )
    print(f"\nV-optimal  J_V = {designer._criterion_value:.6f}")
    designer.print_optimal_candidates(tol=1e-3)
    v_opt_efforts = designer.efforts.copy()

    # ── 4. A-optimal design (comparison) ──────────────────────────────────────
    # A-optimal minimises total parameter variance: J_A = trace(FIM^{-1})
    # This spreads experimental effort to reduce uncertainty in all parameters
    # equally, without regard for where predictions will be used.
    print("\n" + "=" * 60)
    print("COMPARISON: A-optimal design")
    print("=" * 60)

    designer.design_experiment(
        criterion               = designer.a_opt_criterion,
        package                 = "ipopt",
        optimizer               = LINEAR_SOLVER,
        opt_options             = {"tol": 1e-8, "max_iter": 1000},
        optimize_sampling_times = True,
    )
    print(f"\nA-optimal  J_A = {designer._criterion_value:.6f}")
    designer.print_optimal_candidates(tol=1e-3)
    a_opt_efforts = designer.efforts.copy()

    # ── 5. D-optimal design (comparison) ──────────────────────────────────────
    # D-optimal maximises det(FIM), minimising the volume of the joint
    # parameter confidence ellipsoid. It selects experiments that maximise
    # the overall information content without targeting any specific
    # prediction direction. Typically pushes experiments to the extremes
    # of the operating space to maximise sensitivity contrasts.
    print("\n" + "=" * 60)
    print("COMPARISON: D-optimal design")
    print("=" * 60)

    designer.design_experiment(
        criterion               = designer.d_opt_criterion,
        package                 = "ipopt",
        optimizer               = LINEAR_SOLVER,
        opt_options             = {"tol": 1e-8, "max_iter": 1000},
        optimize_sampling_times = True,
    )
    print(f"\nD-optimal  J_D = {designer._criterion_value:.6f}")
    designer.print_optimal_candidates(tol=1e-3)
    d_opt_efforts = designer.efforts.copy()

    # ── 6. Summary: compare J_V across designs ────────────────────────────────
    # Evaluate J_V = trace(W * FIM^{-1} * W^T) for each design using the
    # same W matrix (fixed at dw).  This is the apples-to-apples comparison:
    # how well does each design reduce prediction uncertainty at the point
    # that actually matters for process operation?
    print("\n" + "=" * 60)
    print("SUMMARY: prediction variance at optimal operating point dw")
    print("=" * 60)

    def eval_JV(efforts):
        """Evaluate J_V for a given effort allocation."""
        designer._eval_fim(efforts)
        try:
            fim_inv = np.linalg.inv(designer.fim)
        except np.linalg.LinAlgError:
            fim_inv = np.linalg.pinv(designer.fim)
        return float(np.trace(designer.W @ fim_inv @ designer.W.T))

    jv_vopt = eval_JV(v_opt_efforts)
    jv_aopt = eval_JV(a_opt_efforts)
    jv_dopt = eval_JV(d_opt_efforts)

    jv_dict = {
        'V-optimal': jv_vopt,
        'A-optimal': jv_aopt,
        'D-optimal': jv_dopt,
    }
    efforts_dict = {
        'V-optimal': v_opt_efforts,
        'A-optimal': a_opt_efforts,
        'D-optimal': d_opt_efforts,
    }

    print(f"\n  {'Design':<12}  {'J_V':>12}  {'vs V-optimal':>12}")
    print(f"  {'-' * 40}")
    print(f"  {'V-optimal':<12}  {jv_vopt:>12.6f}  {'(best by construction)':>12}")
    print(f"  {'A-optimal':<12}  {jv_aopt:>12.6f}  "
          f"{f'{jv_aopt/jv_vopt:.2f}x worse':>12}")
    print(f"  {'D-optimal':<12}  {jv_dopt:>12.6f}  "
          f"{f'{jv_dopt/jv_vopt:.2f}x worse':>12}")

    # ── 7. Visualise results ───────────────────────────────────────────────────
    print("\nPlotting design comparison...")
    plot_design_comparison(
        efforts_dict   = efforts_dict,
        tic_candidates = tic_candidates,
        dw_tic         = dw_tic,
        jv_dict        = jv_dict,
        mp             = THETA_GUESS,
    )
    plot_jv_bar(jv_dict)
    plt.show()
