from pro_solver.benchmark.data_modules.load_data import equation_to_numpy
from pro_solver.benchmark.benchmark_modules.advection1d import solve_pde
import numpy as np


def benchmark():
    # загружаем данные в том же формате, что и для rec_diff
    t, x, u_0, target = equation_to_numpy('advection1d')

    # считаем решение для первого примера
    u = solve_pde(x, t, u_0[0])

    # относительная квадратичная ошибка
    rel_loss = np.sum((u - target[0]) ** 2) / np.sum((target[0]) ** 2)
    print(rel_loss)


if __name__ == "__main__":
    benchmark()
