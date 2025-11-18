from pro_solver.benchmark.data_modules.load_data import equation_to_numpy
from pro_solver.benchmark.benchmark_modules.rec_diff2d import solve_pde
import numpy as np


def benchmark():
    # ожидаем формат:
    # t, x, y, u_0, target
    t, x, y, u_0, target = equation_to_numpy('rec_diff2d')

    # считаем решение для первого примера
    # u_0[0] имеет форму (Ny, Nx)
    u = solve_pde(x, y, t, u_0[0])

    # target[0] ожидаем формы (Nt, Ny, Nx)
    rel_loss = np.sum((u - target[0]) ** 2) / np.sum((target[0]) ** 2)
    print(rel_loss)


if __name__ == "__main__":
    benchmark()
