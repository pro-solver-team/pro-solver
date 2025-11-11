from pro_solver.benchmark.data_modules.load_data import equation_to_numpy
from pro_solver.benchmark.benchmark_modules.rec_diff import solve_pde
import numpy as np

def benchmark():
    t, x, u_0, target = equation_to_numpy('rec_diff')
    u = solve_pde(x, t, u_0[0])
    rel_loss = np.sum((u - target[0]) ** 2) / np.sum((target[0]) ** 2)
    print(rel_loss)

if __name__ == "__main__":
    benchmark()