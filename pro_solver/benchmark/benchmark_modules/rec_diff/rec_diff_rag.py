import numpy as np

def solve_pde(x, t, u0):
    Nx, Nt = len(x), len(t)
    dx, dt = x[1] - x[0], t[1] - t[0]
    u = np.zeros((Nt, Nx))
    u[0] = u0
    for n in range(1, Nt):
        u[n, 1:-1] = u[n-1, 1:-1] - 0.5 * dt / dx**2 * (u[n-1, 2:] - 2 * u[n-1, 1:-1] + u[n-1, :-2]) - dt * u[n-1, 1:-1] * (1 - u[n-1, 1:-1])
    return u