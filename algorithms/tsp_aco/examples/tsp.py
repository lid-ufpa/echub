import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import ALPHA, BETA, EVAPORATION_RATE, N_ANTS, N_CITIES, N_ITERATIONS
from src.tsp_aco import TSPACO


def create_tsp(n_cities: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 100, size=(n_cities, 2))
    cost_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(i, n_cities):
            dist = np.linalg.norm(coords[i] - coords[j])
            cost_matrix[i, j] = cost_matrix[j, i] = dist
    return coords, cost_matrix


if __name__ == "__main__":
    city_coords, distance_matrix = create_tsp(n_cities=N_CITIES)

    tsp_aco = TSPACO(
        cost_matrix=distance_matrix,
        n_ants=N_ANTS,
        alpha=ALPHA,
        beta=BETA,
        n_iterations=N_ITERATIONS,
        evaporation_rate=EVAPORATION_RATE,
    )

    best_solution, best_cost = tsp_aco.optimize(start_node=0)
    print(f"✔ Best solution found: {best_solution}")
    print(f"✔ Cost of best solution: {best_cost:.2f}")

    print("→ Generating animation...")
    tsp_aco.animate(coords=city_coords, output_filename="tsp_aco_animation")
    print("✔ Animation saved.")
