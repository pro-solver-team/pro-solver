import numpy as np
from scipy.integrate import solve_ivp
from numba import njit


def solve_pde(x, y, t, u0, nu=0.01, fx=0.0, fy=0.0):
    """
    Solve a simplified 2D incompressible Navier-Stokes-like system
    for velocity field (u, v) with periodic boundary conditions.

        du/dt + u du/dx + v du/dy = nu (d^2u/dx^2 + d^2u/dy^2) + fx
        dv/dt + u dv/dx + v dv/dy = nu (d^2v/dx^2 + d^2v/dy^2) + fy

    This is a 2D viscous advection-diffusion system for (u, v).
    Pressure projection is not implemented, so "incompressibility"
    is only approximate.

    Parameters
    ----------
    x : np.ndarray
        Spatial mesh in x-direction, shape (Nx,)
    y : np.ndarray
        Spatial mesh in y-direction, shape (Ny,)
    t : np.ndarray
        Time mesh, shape (Nt,)
    u0 : np.ndarray
        Initial condition, shape (Ny, Nx, 2), fields [u, v]
    nu : float
        Viscosity coefficient
    fx : float
        Constant body force in x-direction
    fy : float
        Constant body force in y-direction

    Returns
    -------
    u : np.ndarray
        Solution, shape (Nt, Ny, Nx, 2), fields [u, v]
    """
    Ny = len(y)
    Nx = len(x)
    Nt = len(t)

    if u0.ndim != 3 or u0.shape[0] != Ny or u0.shape[1] != Nx or u0.shape[2] != 2:
        raise ValueError(
            f"u0 must have shape (Ny, Nx, 2), got {u0.shape}"
        )

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    y0 = u0.reshape(-1)  # (2 * Ny * Nx,)

    @njit
    def grad_x_periodic(f):
        df = np.empty_like(f)
        # interior
        df[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * dx)
        # periodic boundaries in x
        df[:, 0] = (f[:, 1] - f[:, -1]) / (2.0 * dx)
        df[:, -1] = (f[:, 0] - f[:, -2]) / (2.0 * dx)
        return df

    @njit
    def grad_y_periodic(f):
        df = np.empty_like(f)
        # interior
        df[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2.0 * dy)
        # periodic boundaries in y
        df[0, :] = (f[1, :] - f[-1, :]) / (2.0 * dy)
        df[-1, :] = (f[0, :] - f[-2, :]) / (2.0 * dy)
        return df

    @njit
    def laplacian_periodic(f):
        lf = np.empty_like(f)

        # second derivative in x
        f_xx = np.empty_like(f)
        f_xx[:, 1:-1] = (f[:, 2:] - 2.0 * f[:, 1:-1] + f[:, :-2]) / (dx * dx)
        f_xx[:, 0] = (f[:, 1] - 2.0 * f[:, 0] + f[:, -1]) / (dx * dx)
        f_xx[:, -1] = (f[:, 0] - 2.0 * f[:, -1] + f[:, -2]) / (dx * dx)

        # second derivative in y
        f_yy = np.empty_like(f)
        f_yy[1:-1, :] = (f[2:, :] - 2.0 * f[1:-1, :] + f[:-2, :]) / (dy * dy)
        f_yy[0, :] = (f[1, :] - 2.0 * f[0, :] + f[-1, :]) / (dy * dy)
        f_yy[-1, :] = (f[0, :] - 2.0 * f[-1, :] + f[-2, :]) / (dy * dy)

        lf[:, :] = f_xx + f_yy
        return lf

    @njit
    def pde_rhs(t_local, y_vec):
        # unpack velocities
        u_flat = y_vec[0:Ny * Nx]
        v_flat = y_vec[Ny * Nx:2 * Ny * Nx]

        u = u_flat.reshape((Ny, Nx))
        v = v_flat.reshape((Ny, Nx))

        # gradients
        du_dx = grad_x_periodic(u)
        du_dy = grad_y_periodic(u)
        dv_dx = grad_x_periodic(v)
        dv_dy = grad_y_periodic(v)

        # laplacians
        u_lap = laplacian_periodic(u)
        v_lap = laplacian_periodic(v)

        # convective terms (u · grad) u, (u · grad) v
        conv_u = u * du_dx + v * du_dy
        conv_v = u * dv_dx + v * dv_dy

        # RHS
        u_t = -conv_u + nu * u_lap + fx
        v_t = -conv_v + nu * v_lap + fy

        du_dt_flat = u_t.reshape(-1)
        dv_dt_flat = v_t.reshape(-1)

        dy_vec = np.empty_like(y_vec)
        dy_vec[0:Ny * Nx] = du_dt_flat
        dy_vec[Ny * Nx:2 * Ny * Nx] = dv_dt_flat

        return dy_vec

    sol = solve_ivp(
        pde_rhs,
        (t[0], t[-1]),
        y0,
        t_eval=t,
        method="BDF",
    )

    y_all = sol.y.T  # (Nt, 2 * Ny * Nx)
    u_all = y_all.reshape(Nt, Ny, Nx, 2)

    return u_all
