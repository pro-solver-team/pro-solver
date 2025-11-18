from pro_solver.benchmark.data_modules.load_data import equation_to_numpy
from pro_solver.benchmark.benchmark_modules.shallow_water2d import solve_pde
import numpy as np


def benchmark():
    # Загружаем данные в стиле 2D задач:
    # t, x, y, u_0, target
    t, x, y, u_0, target = equation_to_numpy('shallow_water2d')

    # u_0[0] имеет форму (Ny, Nx, 3): [h, u, v]
    u = solve_pde(x, y, t, u_0[0])

    # target[0] ожидаем формы (Nt, Ny, Nx, 3)
    rel_loss = np.sum((u - target[0]) ** 2) / np.sum((target[0]) ** 2)
    print(rel_loss)


if __name__ == "__main__":
    benchmark()
