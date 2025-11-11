import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

def solve_pde(x, t, u0):
    """
    Solve the PDE du/dt - 0.5 * d^2u/dx^2 - 2 * u * (1-u) = 0.0
    
    Parameters:
    x : np.array
        Spatial mesh with shape (Nx,)
    t : np.array
        Time mesh with shape (Nt,)
    u0 : np.array
        Initial condition with shape (Nx,)
    
    Returns:
    u : np.array
        Solution with shape (Nt, Nx)
    """
    Nx = len(x)
    Nt = len(t)
    
    @njit
    def pde_func(t, u):
        du = np.zeros_like(u)
        du[1:-1] = 0.5 * (u[2:] - 2 * u[1:-1] + u[:-2]) / (x[1] - x[0])**2 - 2 * u[1:-1] * (1 - u[1:-1])
        du[0] = 0.5 * (u[1] - 2 * u[0] + u[-1]) / (x[1] - x[0])**2 - 2 * u[0] * (1 - u[0])
        du[-1] = 0.5 * (u[0] - 2 * u[-1] + u[-2]) / (x[1] - x[0])**2 - 2 * u[-1] * (1 - u[-1])
        return du
    
    sol = solve_ivp(pde_func, [t[0], t[-1]], u0, t_eval=t, method='BDF')
    u = sol.y.T
    return u
