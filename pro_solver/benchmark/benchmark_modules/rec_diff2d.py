import numpy as np
from scipy.integrate import solve_ivp
from numba import njit


@njit
def laplacian_periodic(u_field, dx, dy):
    """
    u_field: (Ny, Nx)
    Возвращает лапласиан с периодическими ГУ по x и y.
    """
    Ny, Nx = u_field.shape
    u_xx = np.empty_like(u_field)
    u_yy = np.empty_like(u_field)

    # d^2u / dx^2, периодика по x
    u_xx[:, 1:-1] = (u_field[:, 2:] - 2.0 * u_field[:, 1:-1] + u_field[:, :-2]) / (dx * dx)
    u_xx[:, 0] = (u_field[:, 1] - 2.0 * u_field[:, 0] + u_field[:, -1]) / (dx * dx)
    u_xx[:, -1] = (u_field[:, 0] - 2.0 * u_field[:, -1] + u_field[:, -2]) / (dx * dx)

    # d^2u / dy^2, периодика по y
    u_yy[1:-1, :] = (u_field[2:, :] - 2.0 * u_field[1:-1, :] + u_field[:-2, :]) / (dy * dy)
    u_yy[0, :] = (u_field[1, :] - 2.0 * u_field[0, :] + u_field[-1, :]) / (dy * dy)
    u_yy[-1, :] = (u_field[0, :] - 2.0 * u_field[-1, :] + u_field[-2, :]) / (dy * dy)

    return u_xx + u_yy


@njit
def rhs_recdiff2d(u_field, dx, dy):
    """
    Правые части для 2D diffusion-reaction:

        u_t = 0.5 * (u_xx + u_yy) + 2 * u * (1 - u)

    u_field: (Ny, Nx), C-contiguous
    """
    lap_u = laplacian_periodic(u_field, dx, dy)
    reaction = 2.0 * u_field * (1.0 - u_field)
    du_dt = 0.5 * lap_u + reaction
    return du_dt


def solve_pde(x, y, t, u0):
    """
    Solve the 2D diffusion-reaction PDE

        du/dt = 0.5 * (d^2u/dx^2 + d^2u/dy^2) + 2 * u * (1 - u)

    with periodic boundary conditions in both x and y.

    Parameters
    ----------
    x : np.ndarray
        Spatial mesh in x direction, shape (Nx,)
    y : np.ndarray
        Spatial mesh in y direction, shape (Ny,)
    t : np.ndarray
        Time mesh, shape (Nt,)
    u0 : np.ndarray
        Initial condition, shape (Ny, Nx)

    Returns
    -------
    u : np.ndarray
        Solution, shape (Nt, Ny, Nx)
    """
    Nx = len(x)
    Ny = len(y)
    Nt = len(t)

    if u0.shape != (Ny, Nx):
        raise ValueError(f"u0 must have shape (Ny, Nx), got {u0.shape}")

    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    y0 = u0.reshape(-1)  # (Ny * Nx,)

    def pde_func(t_local, y_vec):
        """
        Обёртка для solve_ivp:
        SciPy сюда может передавать не-contiguous y_vec, поэтому
        мы аккуратно reshапим и делаем копию перед вызовом jitted-ядра.
        """
        # сначала в 2D
        u_field_view = y_vec.reshape((Ny, Nx))
        # делаем C-contiguous массив для Numba
        u_field = np.ascontiguousarray(u_field_view)
        du_dt = rhs_recdiff2d(u_field, dx, dy)  # (Ny, Nx)
        return du_dt.reshape(-1)

    dt_min = float(np.min(np.diff(t)))

    sol = solve_ivp(
        pde_func,
        (float(t[0]), float(t[-1])),
        y0,
        t_eval=t,
        method="BDF",
        max_step=dt_min,
        rtol=1e-6,
        atol=1e-9,
    )

    u = sol.y.T.reshape((Nt, Ny, Nx))
    return u
