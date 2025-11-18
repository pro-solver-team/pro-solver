from pro_solver.benchmark.data_modules.load_data import equation_to_numpy
from pro_solver.benchmark.benchmark_modules.incomp_ns2d import solve_pde
import numpy as np


def benchmark():
    # грузим данные в том же стиле, что и для 1D задач,
    # только теперь есть ещё y-координата
    t, x, y, u_0, target = equation_to_numpy('incomp_ns2d')

    # u_0[0] имеет форму (Ny, Nx, 2): [u, v]
    u = solve_pde(x, y, t, u_0[0])

    # target[0] ожидаем формы (Nt, Ny, Nx, 2)
    rel_loss = np.sum((u - target[0]) ** 2) / np.sum((target[0]) ** 2)
    print(rel_loss)


if __name__ == "__main__":
    benchmark()
