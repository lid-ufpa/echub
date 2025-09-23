from .benchmarks import ackley, griewank, rastrigin, rosenbrock, schaffer, sphere

N_PARTICLES = 30
ITERATIONS = 100
DIMENSION = 2

INERTIA = 0.5
COGNITIVE = 1.5
SOCIAL = 1.5

FUNCTIONS = {
    "sphere": [sphere, (-5.12, 5.12)],
    "rastrigin": [rastrigin, (-5.12, 5.12)],
    "rosenbrock": [rosenbrock, (-5, 10)],
    "ackley": [ackley, (-32.768, 32.768)],
    "griewank": [griewank, (-600, 600)],
    "schaffer": [schaffer, (-100, 100)],
}

GRAPHS_PATH = "algorithms/pso/results/graphs/"
ANIMATIONS_2D_PATH = "algorithms/pso/results/2d_animations/"
ANIMATIONS_3D_PATH = "algorithms/pso/results/3d_animations/"

FUNCTION_NAMES = ["sphere", "rastrigin", "rosenbrock", "ackley", "griewank", "schaffer"]
