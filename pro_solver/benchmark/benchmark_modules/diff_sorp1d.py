import numpy as np
from scipy.integrate import solve_ivp
from numba import njit


def solve_pde(x, t, u0, D=0.5, k=1.0):
    """
    Solve the 1D diffusion-sorption equation

        du/dt = D * d^2u/dx^2 - k * u

    with periodic boundary conditions.

    Parameters
    ----------
    x : np.array
        Spatial mesh with shape (Nx,)
    t : np.array
        Time mesh with shape (Nt,)
    u0 : np.array
        Initial condition with shape (Nx,)
    D : float
        Diffusion coefficient
    k : float
        Sorption (decay) coefficient

    Returns
    -------
    u : np.array
        Solution with shape (Nt, Nx)
    """
    Nx = len(x)
    Nt = len(t)
    dx = x[1] - x[0]

    @njit
    def pde_func(t_local, u):
        du = np.zeros_like(u)

        # second derivative u_xx with periodic boundaries
        du[1:-1] = D * (u[2:] - 2 * u[1:-1] + u[:-2]) / (dx ** 2) - k * u[1:-1]
        du[0] = D * (u[1] - 2 * u[0] + u[-1]) / (dx ** 2) - k * u[0]
        du[-1] = D * (u[0] - 2 * u[-1] + u[-2]) / (dx ** 2) - k * u[-1]

        return du

    sol = solve_ivp(pde_func, [t[0], t[-1]], u0, t_eval=t, method='BDF')
    u = sol.y.T
    return u
