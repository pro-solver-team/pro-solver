import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def solve_pde(x, t, u0):
    """
    Solves the PDE:
        du/dt - 0.5 * d²u/dx² - 2 * u * (1 - u) = 0
    on x ∈ [0,1], t ∈ (0,1], with periodic BCs.

    Args:
        x: np.ndarray, shape (Nx,), spatial grid.
        t: np.ndarray, shape (Nt,), time grid.
        u0: np.ndarray, shape (Nx,), initial condition u(0, x).

    Returns:
        u: np.ndarray, shape (Nt, Nx), solution on the mesh.
    """
    Nx = x.size
    Nt = t.size
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # Precompute Laplacian matrix (periodic BCs)
    diagonals = [np.ones(Nx) * (-2.0 / dx**2),
                 np.ones(Nx-1) * (1.0 / dx**2),
                 np.ones(Nx-1) * (1.0 / dx**2)]
    L = diags(diagonals, [0, -1, 1], shape=(Nx, Nx)).toarray()
    L[0, -1] = 1.0 / dx**2  # Periodic BC
    L[-1, 0] = 1.0 / dx**2

    # Initialize solution
    u = np.zeros((Nt, Nx))
    u[0] = u0

    # Implicit Euler method (backward)
    I = np.eye(Nx)
    A = I - 0.5 * dt * L

    for n in range(1, Nt):
        u_prev = u[n-1]
        f_nonlinear = 2 * u_prev * (1 - u_prev)
        b = u_prev + dt * f_nonlinear
        u[n] = spsolve(A, b)

    return u
