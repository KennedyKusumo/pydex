# pydex — Development Plan

## Goal

Bring pydex to a state that reflects my current standards: clean architecture, modern Python, well-tested, publishable. The approach is incremental — each phase leaves the project better without breaking the existing API for users.

---

## Phase 1 — Type Hints + Static Analysis

**Why first:** Type hints are the scaffolding for everything else. They make refactoring safer, expose hidden assumptions, and give IDEs and mypy visibility into the codebase. This phase touches every file but doesn't change behaviour.

**Tasks:**
- [x] Add type hints to all public methods in `designer.py`
- [x] Add `numpy.typing` (`NDArray`) throughout
- [x] Add `py.typed` marker file (PEP 561)
- [x] Configure `mypy` in `pyproject.toml`
- [ ] Install mypy + run: `pip install mypy && mypy pydex/`
- [ ] Resolve all mypy errors
- [x] Add type hints to `bnb/node.py`, `bnb/tree.py`, and utils
- [x] Fix `is` vs `==` string comparison bugs in `bnb/node.py`

**Definition of done:** `mypy --strict` passes on the core package.

---

## Phase 2 — Decompose Designer into Focused Classes ✓

**Why second:** The 5,390-line God class is the root cause of most maintainability problems. Splitting it out makes testing, documentation, and future development tractable. Type hints from Phase 1 make this refactor safer.

**Result:** `designer.py` reduced from 5,390 → 1,221 lines across 9 focused mixin modules:

| Mixin | Responsibility |
|---|---|
| `_mixin_init.py` — `DesignerInit` | Properties, `initialize`, candidate enumeration, init validation |
| `_mixin_io.py` — `DesignerIO` | Logging, load/save, result paths |
| `_mixin_criteria.py` — `DesignerCriteria` | D/A/E, prediction-oriented, CVaR criteria |
| `_mixin_visualization.py` — `DesignerVisualization` | All plot methods |
| `_mixin_apportionment.py` — `DesignerApportionment` | `apportion`, Adams algorithm |
| `_mixin_estimability.py` — `DesignerEstimability` | Estimability study, normalize sensitivities |
| `_mixin_estimation.py` — `DesignerEstimation` | `estimate_parameters`, Bayesian PE, residuals |
| `_mixin_sensitivity.py` — `DesignerSensitivity` | Simulate, FIM, sensitivities |
| `_mixin_candidates.py` — `DesignerCandidates` | `get_optimal_candidates`, effort filtering |

**Definition of done:** ✓ All 6 runnable examples pass. `designer.py` contains only `__init__`, `design_experiment`, `solve_cvar_problem`, and their private formulation helpers.

---

## Phase 3 — Proper `simulate()` Interface

**Why:** The current pattern (`designer.simulate = my_function`) is unidiomatic, breaks IDE support, and makes errors hard to trace. Replacing it with an ABC makes the contract explicit and statically verifiable, and eliminates the fragile signature-sniffing in `_handle_simulate_sig`.

### Design

**New file: `pydex/core/simulate.py`** — an ABC hierarchy with one class per system type:

| Class | User signature | System type |
|---|---|---|
| `SimulatorBase` | abstract `__call__(tic, tvc, mp, spt)` | base |
| `StaticSimulator` | `simulate(ti_controls, model_parameters)` | static, TI controls |
| `DynamicTISimulator` | `simulate(ti_controls, sampling_times, model_parameters)` | dynamic, TI only |
| `DynamicTVSimulator` | `simulate(tv_controls, sampling_times, model_parameters)` | dynamic, TV only |
| `DynamicFullSimulator` | `simulate(ti_controls, tv_controls, sampling_times, model_parameters)` | dynamic, both |
| `DynamicUncontrolledSimulator` | `simulate(sampling_times, model_parameters)` | dynamic, no controls |

Each class carries three boolean flags (`is_dynamic`, `has_tv_controls`, `has_ti_controls`) that `initialize()` reads directly — replacing the integer `_simulate_signature` and the `_handle_simulate_sig` sniffing entirely.

**Backward compatibility:** A private `_LegacySimulatorAdapter` wraps bare callables using the old sniffing logic, emitting a `DeprecationWarning`. The existing `designer.simulate = my_function` pattern keeps working via a deprecated property alias on `Designer`. No example changes are needed during PR 1.

### Tasks

**PR 1 — Interface + shim (no example changes):**
- [ ] Create `pydex/core/simulate.py` with `SimulatorBase` and 5 named subclasses
- [ ] Add `_LegacySimulatorAdapter` (private, wraps bare callables with deprecation warning)
- [ ] Add `simulator` property to `Designer`; deprecate `simulate` as alias
- [ ] Replace `_handle_simulate_sig` + `_initialize_internal_simulate_function` with `_configure_simulator()` in `_mixin_init.py`
- [ ] Rewrite `_get_component_sizes` to use the three boolean flags instead of integer `_simulate_signature`
- [ ] Update `_swap_candidates` / `_revert_candidates` in `_mixin_criteria.py` to use `_go_simulator`
- [ ] Make `_simulate_internal` a thin delegating method in `_mixin_sensitivity.py`
- [ ] Export the 5 public classes from `pydex/core/__init__.py`
- [ ] Verify all 6 runnable examples pass (with deprecation warnings, no failures)

**PR 2 — Migrate examples to named classes:**
- [ ] Update all examples in `examples/` to use the appropriate named simulator class
- [ ] Remove `DesignerInit.simulate` stub (no longer needed once all examples migrated)
- [ ] Write `MIGRATION.md` with before/after for each of the 5 signature types

**Definition of done:** All examples use named simulator classes. `_LegacySimulatorAdapter` still exists but is never invoked by any shipped example. mypy catches a wrong `simulate()` signature at type-check time.

---

## Phase 4 — Unit Tests

**Why:** Currently only integration tests exist (running full examples). Individual methods have no coverage. This makes refactoring risky and bug discovery slow.

**Tasks:**
- [ ] Set up `pytest` with `tests/` directory
- [ ] Add `conftest.py` with shared fixtures (simple linear model, sensitivities, FIM)
- [ ] Unit tests for each criterion in `criteria.py`
- [ ] Unit tests for FIM assembly and sensitivity computation
- [ ] Unit tests for apportionment methods
- [ ] Unit tests for parameter estimation
- [ ] Keep existing integration tests as smoke tests
- [ ] Add coverage reporting (`pytest-cov`)

**Target:** >80% coverage on core package.

---

## Phase 5 — Modern Build, CI, and Tooling

**Tasks:**
- [ ] Remove `setup.py` — consolidate into `pyproject.toml` only
- [ ] Complete `pyproject.toml` (dependencies, optional extras, metadata)
- [ ] Add `ruff` for linting and formatting
- [ ] Add `pre-commit` hooks (ruff, mypy)
- [ ] Set up GitHub Actions:
  - [ ] CI: lint + type-check + test on push/PR
  - [ ] Release: publish to PyPI on tag
- [ ] Delete `pydex_todo.docx` and `docs/readthedocs_structure.docx` — move to GitHub Issues / this file
- [ ] Move or remove `examples/unpolished/`

**Python version target:** 3.13 (stable, best dependency compatibility as of early 2026). Revisit 3.14 once key deps (numpy, scipy, cvxpy) confirm stable wheels.

---

## Phase 6 — Documentation

**Tasks:**
- [ ] Consistent NumPy-style docstrings across all public methods
- [ ] Set up Sphinx + ReadTheDocs
- [ ] API reference (auto-generated from docstrings)
- [ ] User guide: quickstart, ODE models, custom criteria
- [ ] Concept guide: what is OED, how pydex implements it
- [ ] Changelog (`CHANGELOG.md`)
- [ ] Contributing guide (`CONTRIBUTING.md`)

---

## Phase 7 — New Features

To be planned after Phases 1–6. Candidates (from original `pydex_todo.docx` and author intent):
- [ ] TBD — capture feature ideas as GitHub Issues

---

## Guiding Principles

- **Each phase is independently deployable** — don't hold a release waiting for phase N+1
- **No breaking the public API without a deprecation cycle**
- **Tests before refactor** — if a method has no test, write one before touching it
- **One concern per module** — if I can't name a module's single responsibility in 5 words, it's doing too much
