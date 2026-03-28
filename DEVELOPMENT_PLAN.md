# pydex — Development Plan

## Goal

Bring pydex to a state that reflects the current author's standards: clean architecture, modern Python, well-tested, publishable. The approach is incremental — each phase leaves the project better without breaking the existing API for users.

---

## Phase 1 — Type Hints + Static Analysis

**Why first:** Type hints are the scaffolding for everything else. They make refactoring safer, expose hidden assumptions, and give IDEs and mypy visibility into the codebase. This phase touches every file but doesn't change behavior.

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

## Phase 2 — Decompose Designer into Focused Classes

**Why second:** The 5,390-line God class is the root cause of most maintainability problems. Splitting it out makes testing, documentation, and future development tractable. Type hints from Phase 1 make this refactor safer.

**Proposed decomposition:**

| New Class / Module | Responsibility | Extracted from |
|---|---|---|
| `designer.py` (slimmed) | Orchestration, user interface, state management | Existing Designer |
| `criteria.py` | All optimality criteria (D, A, E, CVaR, prediction-oriented, pseudo-Bayesian) | Designer.d_opt_criterion, etc. |
| `sensitivity.py` | Sensitivity computation, FIM assembly, atomic FIMs | Designer.eval_sensitivities, eval_fim, etc. |
| `estimation.py` | Parameter estimation (least-squares, MCMC/emcee) | Designer.estimate_parameters, insilico_bayesian_inference |
| `estimability.py` | Parameter estimability analysis | Designer.estimability_study, etc. |
| `visualization.py` | All plot_* methods | Designer.plot_* (20+ methods) |
| `apportionment.py` | Apportion continuous design to integer experiments | Designer.apportion |
| `io.py` | Save/load results (dill-based) | Designer.write_oed_result, load_oed_result |

**Tasks:**
- [ ] Extract criteria methods → `criteria.py`
- [ ] Extract sensitivity/FIM methods → `sensitivity.py`
- [ ] Extract visualization → `visualization.py`
- [ ] Extract estimation → `estimation.py`
- [ ] Extract estimability → `estimability.py`
- [ ] Extract apportionment → `apportionment.py`
- [ ] Extract IO → `io.py`
- [ ] Slim down Designer to orchestration only
- [ ] Ensure all existing examples still run (integration test gate)

**Definition of done:** No single module exceeds ~500 lines. All examples pass.

---

## Phase 3 — Proper `simulate()` Interface

**Why:** The current monkey-patch pattern (`designer.simulate = my_function`) is unidiomatic, breaks IDE support, and makes errors hard to trace. Replace with a Protocol or ABC that users can implement clearly.

**Tasks:**
- [ ] Define `SimulateProtocol` (or abstract base) with typed signature
- [ ] Support both subclassing and callable assignment (backwards compatibility)
- [ ] Update all examples to use the new interface
- [ ] Document migration path

**Definition of done:** Users can subclass `Designer` with a typed `simulate()` method and get full IDE completion.

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
- **One concern per module** — if you can't name a module's single responsibility in 5 words, it's doing too much
