import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    COGNITIVE,
    DIMENSION,
    FUNCTION_NAMES,
    FUNCTIONS,
    INERTIA,
    ITERATIONS,
    N_PARTICLES,
    SOCIAL,
)
from src.pso import PSO


def run(function_name: str, idx: int, total: int) -> None:
    print(f"{'=' * 80}")
    print(f"[{idx}/{total}] Starting optimization for: {function_name}")
    print(f"{'=' * 80}")

    function = FUNCTIONS[function_name][0]
    bounds = FUNCTIONS[function_name][1]

    rng = np.random.default_rng()
    initial_positions = rng.uniform(bounds[0], bounds[1], size=(N_PARTICLES, DIMENSION))

    pso = PSO(
        positions=initial_positions,
        fitness=function,
        iterations=ITERATIONS,
    )
    gbest_position, gbest = pso.optimize(INERTIA, COGNITIVE, SOCIAL)

    print(f"✔ Best solution found: {gbest_position}")
    print(f"✔ Fitness of best solution: {gbest}")

    print("→ Generating convergence graph...")
    pso.plot_convergence(
        f"{function_name}_graph",
        function_name,
    )
    print("✔ Convergence graph saved.")

    animation_dimension = DIMENSION
    if animation_dimension == DIMENSION:
        print("→ Generating 2D animation...")
        pso.animate_2d(
            output_filename=f"{function_name}_animation_2d",
            function_name=function_name,
            bounds=bounds,
        )
        print("✔ 2D animation saved.")

        print("→ Generating 3D animation...")
        pso.animate_3d(
            output_filename=f"{function_name}_animation_3d",
            function_name=function_name,
            bounds=bounds,
        )
        print("✔ 3D animation saved.")

    print(f"✔ Finished {function_name}\n")


if __name__ == "__main__":
    total = len(FUNCTION_NAMES)

    for idx, name in enumerate(FUNCTION_NAMES, start=1):
        run(name, idx, total)
