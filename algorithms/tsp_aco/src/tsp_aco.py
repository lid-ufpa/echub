import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from src.config import ANIMATION_PATH


class TSPACO:
    def __init__(
        self,
        cost_matrix: np.ndarray,
        n_ants: int,
        alpha: float,
        beta: float,
        n_iterations: int,
        evaporation_rate: float,
    ) -> None:
        self.cost_matrix = cost_matrix
        self.pheromone_matrix = np.ones(cost_matrix.shape) / len(cost_matrix)
        self.nodes = range(len(cost_matrix))

        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate

        self.best_solution = []
        self.best_solution_cost = float("inf")

        self.history = []

    def _build_solution(self, start_node: int) -> tuple[list[tuple[int, int]], float]:
        solution = []
        visited = {start_node}
        current_node = start_node
        for _ in range(len(self.cost_matrix) - 1):
            candidates = [node for node in self.nodes if node not in visited]
            if not candidates:
                continue
            probabilities = self._transition_probabilities(current_node, candidates)
            rng = np.random.default_rng()
            next_node = rng.choice(candidates, p=probabilities)
            solution.append((current_node, next_node))
            visited.add(next_node)
            current_node = next_node
        solution.append((current_node, start_node))
        total_cost = float(sum(self.cost_matrix[i][j] for i, j in solution))
        return solution, total_cost

    def _build_solutions(self, start_node: int) -> list:
        solutions = []
        for _ in range(self.n_ants):
            solution, cost = self._build_solution(start_node)
            solutions.append((solution, cost))
            self._update_global_best(solution, cost)
        return solutions

    def _transition_probabilities(self, current_node: int, candidates: list) -> list:
        numerators = []
        for j in candidates:
            tau = self.pheromone_matrix[current_node][j] ** self.alpha
            eta = (1 / self.cost_matrix[current_node][j]) ** self.beta
            numerators.append(tau * eta)
        denominator = sum(numerators)
        if denominator == 0:
            return [1 / len(candidates)] * len(candidates)
        return [num / denominator for num in numerators]

    def _update_pheromones(self, solutions: list) -> None:
        delta_pheromones = np.zeros_like(self.pheromone_matrix, dtype=float)
        for solution, cost in solutions:
            for i, j in solution:
                delta_pheromones[i, j] += 1 / cost
        self.pheromone_matrix = (
            (1 - self.evaporation_rate) * self.pheromone_matrix
        ) + delta_pheromones

    def _update_global_best(self, solution: list, cost: float) -> None:
        if cost < self.best_solution_cost:
            self.best_solution_cost = cost
            self.best_solution = solution

    def optimize(self, start_node: int) -> tuple[list, float]:
        self.history = []
        for i in range(self.n_iterations):
            solutions = self._build_solutions(start_node)
            self._update_pheromones(solutions)

            history_entry = {
                "iteration": i,
                "best_solution": self.best_solution.copy(),
                "best_cost": self.best_solution_cost,
                "pheromones": self.pheromone_matrix.copy(),
            }
            self.history.append(history_entry)

        best_solution_nodes = [int(i) for i, j in self.best_solution] + [start_node]

        return best_solution_nodes, self.best_solution_cost

    def animate(self, coords: np.ndarray, output_filename: str) -> None:
        if not self.history:
            return

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(coords[:, 0], coords[:, 1], c="black", s=150, zorder=5)
        for i, (x, y) in enumerate(coords):
            ax.text(
                x,
                y,
                str(i),
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

        pheromone_lines = []
        (best_path_line,) = ax.plot(
            [],
            [],
            color="#E6550D",
            linewidth=2.5,
            zorder=10,
            label="Best Path",
        )

        plt.xlabel("X")
        plt.ylabel("Y")
        ax.legend()

        def update(frame: int) -> list:
            nonlocal pheromone_lines
            for line in pheromone_lines:
                line.remove()
            pheromone_lines = []

            state = self.history[frame]
            pheromones = state["pheromones"]
            norm = pheromones / pheromones.max() if pheromones.max() > 0 else pheromones

            for i in range(len(coords)):
                for j in range(i, len(coords)):
                    lw = norm[i, j] * 7
                    bound = 0.1
                    if lw > bound:
                        (line,) = ax.plot(
                            [coords[i, 0], coords[j, 0]],
                            [coords[i, 1], coords[j, 1]],
                            "#3182BD",
                            alpha=0.5,
                            linewidth=lw,
                            zorder=1,
                        )
                        pheromone_lines.append(line)

            if state["best_solution"]:
                path_coords = np.array(
                    [coords[i] for i, j in state["best_solution"]]
                    + [coords[state["best_solution"][0][0]]],
                )
                best_path_line.set_data(path_coords[:, 0], path_coords[:, 1])

            ax.set_title(
                f"Optimization - Iteration {state['iteration'] + 1}/{self.n_iterations} | "
                f"Best Cost: {state['best_cost']:.2f}",
            )

            return [best_path_line, *pheromone_lines]

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(self.history),
            interval=200,
            blit=True,
            repeat=False,
        )
        ani.save(f"{ANIMATION_PATH}{output_filename}.gif", writer="pillow", fps=5)
        plt.close(fig)
