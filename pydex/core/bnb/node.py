from __future__ import annotations

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from typing import cast


class Node:
    def __init__(
        self,
        int_var: cp.Variable,
        cvxpy_prob: cp.Problem,
        node_id: str | None = None,
        optimizer: str | None = None,
    ) -> None:
        # node identifier
        if node_id is None:
            node_id = "0"
        self.node_id: str = str(node_id)

        # core attributes
        self.int_var: cp.Variable = int_var
        self.cvxpy_prob: cp.Problem = cvxpy_prob

        # computed and stored values
        self.ub: float | None = None
        self.lb: float | None = None
        self.int_var_val: NDArray[np.float64] | None = None

        # flags
        self.solved: bool = False
        self.feasible: bool | None = None
        self.is_incumbent: bool | None = None
        self.worthwhile: bool | None = None
        self.integral: bool | None = None
        self._integrity_tol: int = 4

        # branches
        self.left_child: Node | None = None
        self.right_child: Node | None = None

        # options
        self.optimizer: str | None = optimizer

    def solve(self) -> None:
        if self.solved:
            return

        if self.optimizer is None:
            self.optimizer = cp.MOSEK  # TODO Phase 5: replace with cp.CLARABEL

        self.cvxpy_prob.solve(solver=self.optimizer)
        if self.cvxpy_prob.status == "optimal":
            self.feasible = True
            self.solved = True
            self.ub = self.cvxpy_prob.value
            self.check_integrity()
            relaxed_val = cast(NDArray[np.float64], self.int_var.value)  # non-None after optimal solve
            self.int_var_val = relaxed_val
            if self.integral:
                self.lb = self.ub
            else:
                prob = cp.Problem(
                    self.cvxpy_prob.objective,
                    self.cvxpy_prob.constraints +
                    [self.int_var == np.abs(np.round(relaxed_val))]
                )
                prob.solve()
                if prob.status == "optimal":
                    self.lb = prob.value
                elif prob.status == "infeasible":
                    self.lb = -np.inf

        if self.cvxpy_prob.status == "infeasible":
            self.feasible = False
            self.solved = True

    def check_integrity(self) -> bool:
        assert self.int_var.value is not None, "check_integrity called before solve"
        int_var_val = np.round(self.int_var.value, self._integrity_tol)
        fractional, integral = np.modf(int_var_val)
        if np.allclose(fractional, 0):
            self.integral = True
        else:
            self.integral = False
        return self.integral  # type: ignore[return-value]

    def branch(self, scheme: str = "greatest_fractional") -> tuple[Node, Node]:
        if scheme == "greatest_fractional":
            self._greatest_fractional_branch()
        else:
            raise SyntaxError(
                f"Unknown scheme: {scheme}. try \"greatest_fractional\""
            )
        return self.left_child, self.right_child  # type: ignore[return-value]

    def _greatest_fractional_branch(self) -> None:
        assert self.int_var_val is not None, "_greatest_fractional_branch called before solve"
        # determine the variable to branch over
        if self.int_var.ndim > 1:
            self.int_var = self.int_var.flatten()
        fractional, integral = np.modf(self.int_var_val)
        most_fractional_idx = np.abs(fractional - 0.5).argmin()
        most_fractional_var = self.int_var[most_fractional_idx]
        # creating left and right child nodes
        right_child_cons = [
            most_fractional_var >= np.ceil(self.int_var_val.flatten()[most_fractional_idx])
        ]
        left_bound = np.floor(self.int_var_val.flatten()[most_fractional_idx])
        if np.isclose(left_bound, 0):
            left_child_cons = [
                most_fractional_var == 0
            ]
        else:
            left_child_cons = [
                most_fractional_var <= left_bound
            ]
        # creating children
        self.left_child = Node(
            self.int_var,
            cp.Problem(
                self.cvxpy_prob.objective,
                self.cvxpy_prob.constraints + left_child_cons,
            ),
            self.node_id + ".0"
        )
        self.right_child = Node(
            self.int_var,
            cp.Problem(
                self.cvxpy_prob.objective,
                self.cvxpy_prob.constraints + right_child_cons
            ),
            self.node_id + ".1"
        )

    def __str__(self) -> str:
        width = 80
        if not self.solved:
            return "unsolved node"
        elif not self.feasible:
            return f"[Node {self.node_id}: infeasible]".center(width, "=")
        elif self.integral:
            int_vars_str = str(np.round(self.int_var_val, 2)) if self.int_var_val is not None else "?"
            return (
                f"[Node {self.node_id}: integral solution]".center(width, "=")
                + f"\nUpper bound: {self.ub}"
                + f"\nLower bound: {self.lb}"
                + f"\nInteger Vars: {int_vars_str}"
                + f"\n" + "".center(width, ".")
            )
        else:
            int_vars_str = str(np.round(self.int_var_val, 2)) if self.int_var_val is not None else "?"
            return (
                f"[Node {self.node_id}: non-integral solution]".center(width, "=")
                + f"\nUpper bound: {self.ub}"
                + f"\nLower bound: {self.lb}"
                + f"\nInteger Vars: {int_vars_str}"
                + f"\n" + "".center(width, ".")
            )
