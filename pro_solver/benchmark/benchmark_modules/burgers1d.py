import numpy as np
from scipy.integrate import solve_ivp
from numba import njit


def solve_pde(x, t, u0, nu=0.1):
    """
    Solve the 1D viscous Burgers' equation

        du/dt + u * du/dx - nu * d^2u/dx^2 = 0

    with periodic boundary conditions.

    Parameters
    ----------
    x : np.array
        Spatial mesh with shape (Nx,)
    t : np.array
        Time mesh with shape (Nt,)
    u0 : np.array
        Initial condition with shape (Nx,)
    nu : float
        Viscosity coefficient

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

        u_x = np.zeros_like(u)
        u_x[1:-1] = (u[2:] - u[:-2]) / (2.0 * dx)
        u_x[0] = (u[1] - u[-1]) / (2.0 * dx)
        u_x[-1] = (u[0] - u[-2]) / (2.0 * dx)

        u_xx = np.zeros_like(u)
        u_xx[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx ** 2)
        u_xx[0] = (u[1] - 2.0 * u[0] + u[-1]) / (dx ** 2)
        u_xx[-1] = (u[0] - 2.0 * u[-1] + u[-2]) / (dx ** 2)

        du[:] = -u * u_x + nu * u_xx

        return du

    sol = solve_ivp(pde_func, [t[0], t[-1]], u0, t_eval=t, method='BDF')
    u = sol.y.T
    return u
