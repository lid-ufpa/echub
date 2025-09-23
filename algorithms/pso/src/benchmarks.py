import numpy as np


def sphere(x: np.ndarray) -> float:
    return np.sum(x**2)


def schaffer(x: np.ndarray) -> float:
    total = 0
    for i in range(len(x) - 1):
        num = np.sin(x[i] ** 2 - x[i + 1] ** 2) ** 2 - 0.5
        den = (1 + 0.001 * (x[i] ** 2 + x[i + 1] ** 2)) ** 2
        total += 0.5 + num / den
    return total


def griewank(x: np.ndarray) -> float:
    sum_part = np.sum(x**2) / 4000
    prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + sum_part - prod_part


def ackley(x: np.ndarray) -> float:
    return (
        -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
        - np.exp(np.mean(np.cos(2 * np.pi * x)))
        + 20
        + np.e
    )


def rastrigin(x: np.ndarray) -> float:
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def rosenbrock(x: np.ndarray) -> float:
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
