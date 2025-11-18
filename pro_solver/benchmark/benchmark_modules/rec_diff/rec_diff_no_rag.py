import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.integrate import odeint

def solve_pde(x, t, u0):
    Nx = len(x)
    Nt = len(t)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # Construct Laplacian matrix with periodic BCs
    diag_main = -2 * np.ones(Nx)
    diag_off = np.ones(Nx - 1)
    diag_off[0] = 0  # No connection between first and last (handled separately)
    L = diags([diag_main, diag_off, diag_off], [0, -1, 1]).toarray()
    L[0, -1] = 1  # Periodic BC: u(0) = u(Nx)
    L[-1, 0] = 1  # Periodic BC: u(Nx) = u(0)
    L = 0.5 * L / (dx**2)

    # ODE system: du/dt = L @ u + 2 * u * (1 - u)
    def rhs(u, t, L):
        return L @ u + 2 * u * (1 - u)

    # Solve ODE system
    u = odeint(rhs, u0, t, args=(L,), tcrit=None, mxstep=5000)
    return u
