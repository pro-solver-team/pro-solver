from pro_solver.benchmark.data_modules.load_data import equation_to_numpy
from pro_solver.benchmark.benchmark_modules.rec_diff.rec_diff_rag import solve_pde
from pro_solver.benchmark.benchmark_modules.loss_utils import rel_rmse
from pro_solver.benchmark.benchmark_modules.equation_visualization import plot_evolution
import numpy as np

def benchmark():
    t, x, u_0, target = equation_to_numpy('rec_diff')
    solution = np.empty_like(target)
    for num, u_init in enumerate(u_0):
        u = solve_pde(x=x, t=t, u0=u_init)
        solution[num] = u
    plot_evolution(target[0], path='rag_results/true.png')
    plot_evolution(solution[0], path='rag_results/sol.png')
    plot_evolution(solution[0] - target[0],  path='rag_results/diff.png')
    tar = np.mean(rel_rmse(solution, target))
    print(tar)
    #delta = ( (1/(time_num_steps * x_num_steps)) * ( np.sum(np.sum((u - target) ** 2, axis=-1), axis=-1) ) ) ** 0.5
    #print(delta, tar)
    #rel_loss = np.sum((u[0, 66] - target[0, 66]) ** 2) / np.sum((target[0, 66]) ** 2)
    #print(rel_loss, u.shape)

if __name__ == "__main__":
    benchmark()