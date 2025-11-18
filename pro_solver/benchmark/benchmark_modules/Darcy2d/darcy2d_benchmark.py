from pro_solver.benchmark.data_modules.load_data import equation_to_numpy
from pro_solver.benchmark.benchmark_modules.Darcy2d.darcy2d import solve_pde
from pro_solver.benchmark.benchmark_modules.loss_utils import rel_rmse
from pro_solver.benchmark.benchmark_modules.equation_visualization import plot_evolution
import numpy as np

def benchmark():
    features, x, y, target = equation_to_numpy('darcy2d')
    solution = np.empty_like(target)
    for num, k in enumerate(features):
        u = solve_pde(x, y, k)
        solution[num] = u
    plot_evolution(target[0], path='rag_results/true.png')
    plot_evolution(solution[0], path='rag_results/sol.png')
    plot_evolution(solution[0] - target[0], path='rag_results/diff.png')
    tar = np.mean(rel_rmse(solution, target))
    print(tar)
if __name__ == "__main__":
    benchmark()