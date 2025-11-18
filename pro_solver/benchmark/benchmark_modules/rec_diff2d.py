import numpy as np
from scipy.integrate import solve_ivp
from numba import njit


def solve_pde(x, y, t, u0):
    """
    Solve the 2D diffusion-reaction PDE

        du/dt = 0.5 * (d^2u/dx^2 + d^2u/dy^2) + 2 * u * (1 - u)

    with periodic boundary conditions in both x and y.

    Parameters
    ----------
    x : np.array
        Spatial mesh in x direction with shape (Nx,)
    y : np.array
        Spatial mesh in y direction with shape (Ny,)
    t : np.array
        Time mesh with shape (Nt,)
    u0 : np.array
        Initial condition with shape (Ny, Nx)

    Returns
    -------
    u : np.array
        Solution with shape (Nt, Ny, Nx)
    """
    Nx = len(x)
    Ny = len(y)
    Nt = len(t)

    if u0.shape != (Ny, Nx):
        raise ValueError(f"u0 must have shape (Ny, Nx), got {u0.shape}")

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    y0 = u0.reshape(-1)  # (Ny * Nx,)

    @njit
    def laplacian_periodic(u_field):
        # u_field: (Ny, Nx)
        u_xx = np.empty_like(u_field)
        u_yy = np.empty_like(u_field)

        # second derivative in x
        u_xx[:, 1:-1] = (u_field[:, 2:] - 2.0 * u_field[:, 1:-1] + u_field[:, :-2]) / (dx * dx)
        u_xx[:, 0] = (u_field[:, 1] - 2.0 * u_field[:, 0] + u_field[:, -1]) / (dx * dx)
        u_xx[:, -1] = (u_field[:, 0] - 2.0 * u_field[:, -1] + u_field[:, -2]) / (dx * dx)

        # second derivative in y
        u_yy[1:-1, :] = (u_field[2:, :] - 2.0 * u_field[1:-1, :] + u_field[:-2, :]) / (dy * dy)
        u_yy[0, :] = (u_field[1, :] - 2.0 * u_field[0, :] + u_field[-1, :]) / (dy * dy)
        u_yy[-1, :] = (u_field[0, :] - 2.0 * u_field[-1, :] + u_field[-2, :]) / (dy * dy)

        return u_xx + u_yy

    @njit
    def pde_func(t_local, y_vec):
        u_field = y_vec.reshape((Ny, Nx))

        lap_u = laplacian_periodic(u_field)

        # reaction term: 2 * u * (1 - u)
        reaction = 2.0 * u_field * (1.0 - u_field)

        du_dt = 0.5 * lap_u + reaction

        return du_dt.reshape(-1)

    sol = solve_ivp(
        pde_func,
        [t[0], t[-1]],
        y0,
        t_eval=t,
        method='BDF',
    )

    u = sol.y.T.reshape((Nt, Ny, Nx))
    return u
