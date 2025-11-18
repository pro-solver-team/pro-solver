import numpy as np
from scipy.integrate import solve_ivp
from numba import njit


def solve_pde(x, t, u0, c=1.0):
    """
    Solve the 1D advection equation

        du/dt + c * du/dx = 0

    with periodic boundary conditions.

    Parameters
    ----------
    x : np.array
        Spatial mesh with shape (Nx,)
    t : np.array
        Time mesh with shape (Nt,)
    u0 : np.array
        Initial condition with shape (Nx,)
    c : float
        Advection speed

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

        # upwind схема для du/dx с периодическими границами
        # du/dt = -c * du/dx
        du_dx = np.zeros_like(u)

        # внутренняя часть: u_x ≈ (u[i] - u[i-1]) / dx
        du_dx[1:] = (u[1:] - u[:-1]) / dx
        du_dx[0] = (u[0] - u[-1]) / dx

        du[:] = -c * du_dx
        return du

    sol = solve_ivp(pde_func, [t[0], t[-1]], u0, t_eval=t, method='BDF')
    u = sol.y.T
    return u
