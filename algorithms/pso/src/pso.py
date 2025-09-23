from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


class PSO:
    def __init__(
        self,
        positions: np.ndarray,
        fitness: Callable[[np.ndarray], float],
        iterations: int,
    ) -> None:
        self.fitness = fitness
        self.iterations = iterations
        self.positions = positions
        self.velocities = np.zeros_like(self.positions)
        self.pbest = self.positions.copy()

        best_idx = np.argmin([self.fitness(pb) for pb in self.pbest])
        self.gbest = self.pbest[best_idx].copy()

        self.gbest_history = []
        self.positions_history = []
        self.num_function_call = [0]

    def _update_velocities(
        self,
        inertia: float,
        cognitive: float,
        social: float,
    ) -> None:
        rng = np.random.default_rng()
        r1 = rng.uniform(0, 1, size=self.positions.shape)
        r2 = rng.uniform(0, 1, size=self.positions.shape)

        self.velocities = (
            inertia * self.velocities
            + cognitive * r1 * (self.pbest - self.positions)
            + social * r2 * (self.gbest - self.positions)
        )

    def _update_positions(self) -> None:
        self.positions += self.velocities

    def _update_best_positions(self) -> None:
        for i in range(self.positions.shape[0]):
            if self.fitness(self.positions[i]) < self.fitness(self.pbest[i]):
                self.pbest[i] = self.positions[i].copy()
            if self.fitness(self.positions[i]) < self.fitness(self.gbest):
                self.gbest = self.positions[i].copy()

    def optimize(
        self,
        inertia: float,
        cognitive: float,
        social: float,
    ) -> tuple[np.ndarray, float]:
        for _ in range(self.iterations):
            self._update_velocities(inertia, cognitive, social)
            self._update_positions()
            self._update_best_positions()

            self.positions_history.append(self.positions.copy())

            self.num_function_call.append(self.num_function_call[-1] + 1)
            self.gbest_history.append(self.fitness(self.gbest))
        return self.gbest, self.fitness(self.gbest)

    def plot_convergence(self, output_path: str, function_name: str) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(self.num_function_call[:-1], self.gbest_history)
        plt.title(f"Convergence ({function_name})")
        plt.xlabel("Number of Function Evaluations")
        plt.ylabel("Best Fitness (gbest)")
        plt.savefig(output_path)
        plt.close()

    def generate_animation_2d(
        self,
        output_path: str,
        function_name: str,
        bounds: tuple[float, float],
    ) -> None:
        plt.style.use("default")

        fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")

        x = np.linspace(bounds[0], bounds[1], 200)
        y = np.linspace(bounds[0], bounds[1], 200)
        x_grid, y_grid = np.meshgrid(x, y)
        z_grid = np.array(
            [
                [self.fitness(np.array([x, y])) for x, y in zip(row_x, row_y, strict=False)]
                for row_x, row_y in zip(x_grid, y_grid, strict=False)
            ],
        )
        ax.contour(x_grid, y_grid, z_grid, levels=20, cmap="viridis", linewidths=1.2)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_facecolor("white")

        frames = []
        for i, positions in enumerate(self.positions_history):
            scatter = ax.scatter(
                positions[:, 0],
                positions[:, 1],
                c="#FF8800",
                s=40,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )
            title = ax.text(
                0.5,
                1.05,
                f"Optimization ({function_name}) | Iteration: {i + 1}",
                transform=ax.transAxes,
                ha="center",
                fontsize=14,
                color="black",
            )
            frames.append([scatter, title])

        ani = animation.ArtistAnimation(fig, frames, interval=150, blit=True, repeat_delay=1000)
        ani.save(output_path, writer="pillow")
        plt.close(fig)

    def generate_animation_3d(
        self,
        output_path: str,
        function_name: str,
        bounds: tuple[float, float],
    ) -> None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        x = np.linspace(bounds[0], bounds[1], 50)
        y = np.linspace(bounds[0], bounds[1], 50)
        x_grid, y_grid = np.meshgrid(x, y)
        z = np.array(
            [
                self.fitness(np.array([x, y]))
                for x, y in zip(np.ravel(x_grid), np.ravel(y_grid), strict=False)
            ],
        )
        z_grid = z.reshape(x_grid.shape)

        ax.plot_surface(
            x_grid,
            y_grid,
            z_grid,
            cmap="viridis",
            alpha=0.6,
            rstride=1,
            cstride=1,
            edgecolor="none",
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Fitness")
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)

        frames = []
        for i, positions in enumerate(self.positions_history):
            fitness = np.array([self.fitness(p) for p in positions])

            scatter = ax.scatter(
                positions[:, 0],
                positions[:, 1],
                fitness.astype(float).tolist(),
                c="#FF8800",
                s=40,
                edgecolors="black",
                linewidths=0.5,
                depthshade=True,
            )
            title = ax.text2D(
                0.5,
                0.95,
                f"Optimization ({function_name}) - Iteration: {i + 1}",
                transform=ax.transAxes,
                ha="center",
                fontsize=14,
            )
            frames.append([scatter, title])

        ani = animation.ArtistAnimation(fig, frames, interval=150, blit=True, repeat_delay=1000)
        ani.save(output_path, writer="pillow")
        plt.close(fig)
