from __future__ import annotations

from time import time

import numpy as np

from pydex.core.bnb.node import Node


class Tree:
    def __init__(self, root_node: Node) -> None:
        self.active_nodes: list[Node] = [root_node]
        # end-point nodes
        self.integral_nodes: list[Node] = []
        self.infeasible_nodes: list[Node] = []

        self.selected_node: Node | None = None
        self.optimal_node: Node | None = None

        self.ub: float | None = None
        self.lb: float | None = None

        self.finished: bool = False
        self._verbose: int = 0

    def solve(self) -> Node | None:
        start = time()
        iteration = 1
        print(f"[Branch and Bound]".center(100, "="))
        print(f"# of integer variables  : {self.active_nodes[0].int_var.size}")
        while not self.finished:
            iter_start_time = time()
            if self._verbose >= 2:
                print(f"[Iteration {iteration}: {iter_start_time - start:.2f} seconds]".center(100, "-"))
                print(f"Solving unsolved active nodes...")
            # check if no more active nodes; terminate if yes
            if not self.active_nodes:
                self.finished = True
                for node in self.integral_nodes:
                    assert node.ub is not None
                    if self.ub is None or self.ub <= node.ub:
                        self.ub = node.ub
                        self.optimal_node = node
                        return self.optimal_node
                return self.optimal_node

            # solve active nodes (nodes are smart to not re-solve when already solved)
            for node in self.active_nodes:
                node.solve()
                if self._verbose >= 3:
                    print(f"[Node {node.node_id}]".center(80, "-"))
                    print(f"Upper bound: {node.ub}")
                    print(f"Lower bound: {node.lb}")

            # classify active nodes
            for node in self.active_nodes:
                if not node.feasible:
                    if self._verbose >= 2:
                        print(f"Found infeasible node")
                    self.infeasible_nodes.append(node)
                elif node.integral:
                    if self._verbose >= 2:
                        print(f"Found integral node     : {node.ub}")
                    self.integral_nodes.append(node)
            # remove active nodes that have been classified from self.active_nodes
            self.active_nodes = [node for node in self.active_nodes if node.feasible]
            self.active_nodes = [node for node in self.active_nodes if not node.integral]

            # select most promising active node
            if self.active_nodes:
                self.selected_node = self.active_nodes[
                    np.argmax([node.ub for node in self.active_nodes])  # type: ignore[arg-type]
                ]
            else:
                self.selected_node = self.integral_nodes[
                    np.argmax([node.ub for node in self.integral_nodes])  # type: ignore[arg-type]
                ]

            assert self.selected_node is not None
            if self._verbose >= 2:
                print(f"Complete in {time() - iter_start_time:.2f} CPU seconds.")
                print(f"".center(100, "."))
                print(f"# of active nodes       : {len(self.active_nodes)}")
                print(f"# of integral nodes     : {len(self.integral_nodes)}")
                print(f"# of infeasible nodes   : {len(self.infeasible_nodes)}")
                print(f"Tightest upper bound    : {self.selected_node.ub}")
                if len(self.integral_nodes) != 0:
                    ubs = [node.ub for node in self.integral_nodes if node.ub is not None]
                    print(f"Best integer solution   : {max(ubs) if ubs else 'n/a'}")

            # check if there exist an integral node better than most promising node
            # terminate if yes, with said integral node being optimal.
            assert self.selected_node.ub is not None
            for node in self.integral_nodes:
                assert node.ub is not None
                if node.ub >= self.selected_node.ub:
                    print("".center(100, "-"))
                    print(
                        f"Best integer solution better than tightest upper bound, "
                        f"solution found.".center(100, " ")
                    )
                    print(f"[Finished: {time() - start:.2f} CPU seconds]".center(100, "="))
                    self.finished = True
                    self.optimal_node = node
                    return self.optimal_node

            # branch node with greatest upper bound (most promising)
            # if two nodes have same upper bound, the one generated first is selected
            if self._verbose >= 2:
                print(f"Branching most promising node...")
            left, right = self.selected_node.branch()
            self.active_nodes.extend([left, right])
            self.active_nodes.remove(self.selected_node)

            iteration += 1

        return self.optimal_node
