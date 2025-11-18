from pro_solver.benchmark.data_modules.load_data import equation_to_numpy
from pro_solver.benchmark.benchmark_modules.compressible_ns1d import solve_pde
import numpy as np


def benchmark():
    # Загружаем данные в том же интерфейсе, что и для других 1D задач
    t, x, u_0, target = equation_to_numpy('compressible_ns1d')

    # Берем первое начальное условие и считаем эволюцию
    # u_0[0] имеет форму (Nx, 3): [rho, u, p]
    u = solve_pde(x, t, u_0[0])

    # target[0] должен иметь форму (Nt, Nx, 3)
    rel_loss = np.sum((u - target[0]) ** 2) / np.sum((target[0]) ** 2)
    print(rel_loss)


if __name__ == "__main__":
    benchmark()
