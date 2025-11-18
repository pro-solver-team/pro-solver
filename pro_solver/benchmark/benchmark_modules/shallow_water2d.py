import numpy as np
from scipy.integrate import solve_ivp
from numba import njit


def solve_pde(x, y, t, u0, g=9.81, nu=0.0):
    """
    Solve a simplified 2D shallow-water system for primitive variables (h, u, v)
    with periodic boundary conditions.

        h_t + (h u)_x + (h v)_y = 0
        u_t + u u_x + v u_y + g h_x = nu (u_xx + u_yy)
        v_t + u v_x + v v_y + g h_y = nu (v_xx + v_yy)

    Parameters
    ----------
    x : np.ndarray
        Spatial mesh in x-direction, shape (Nx,)
    y : np.ndarray
        Spatial mesh in y-direction, shape (Ny,)
    t : np.ndarray
        Time mesh, shape (Nt,)
    u0 : np.ndarray
        Initial condition, shape (Ny, Nx, 3), fields [h, u, v]
    g : float
        Gravity acceleration
    nu : float
        Viscosity coefficient

    Returns
    -------
    u : np.ndarray
        Solution, shape (Nt, Ny, Nx, 3), fields [h, u, v]
    """
    Ny = len(y)
    Nx = len(x)
    Nt = len(t)

    if u0.ndim != 3 or u0.shape[0] != Ny or u0.shape[1] != Nx or u0.shape[2] != 3:
        raise ValueError(
            f"u0 must have shape (Ny, Nx, 3), got {u0.shape}"
        )

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    y0 = u0.reshape(-1)  # (3 * Ny * Nx,)

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
        # unpack primitive variables
        h_flat = y_vec[0:Ny * Nx]
        u_flat = y_vec[Ny * Nx:2 * Ny * Nx]
        v_flat = y_vec[2 * Ny * Nx:3 * Ny * Nx]

        h = h_flat.reshape((Ny, Nx))
        u = u_flat.reshape((Ny, Nx))
        v = v_flat.reshape((Ny, Nx))

        # fluxes for continuity
        hu = h * u
        hv = h * v
        hu_x = grad_x_periodic(hu)
        hv_y = grad_y_periodic(hv)

        h_t = -(hu_x + hv_y)

        # gradients of h and velocities
        h_x = grad_x_periodic(h)
        h_y = grad_y_periodic(h)

        u_x = grad_x_periodic(u)
        u_y = grad_y_periodic(u)
        v_x = grad_x_periodic(v)
        v_y = grad_y_periodic(v)

        # laplacians
        u_lap = laplacian_periodic(u)
        v_lap = laplacian_periodic(v)

        # convective terms (u · grad) u, (u · grad) v
        conv_u = u * u_x + v * u_y
        conv_v = u * v_x + v * v_y

        # momentum equations in primitive form
        u_t = -conv_u - g * h_x + nu * u_lap
        v_t = -conv_v - g * h_y + nu * v_lap

        dh_dt_flat = h_t.reshape(-1)
        du_dt_flat = u_t.reshape(-1)
        dv_dt_flat = v_t.reshape(-1)

        dy_vec = np.empty_like(y_vec)
        dy_vec[0:Ny * Nx] = dh_dt_flat
        dy_vec[Ny * Nx:2 * Ny * Nx] = du_dt_flat
        dy_vec[2 * Ny * Nx:3 * Ny * Nx] = dv_dt_flat

        return dy_vec

    sol = solve_ivp(
        pde_rhs,
        (t[0], t[-1]),
        y0,
        t_eval=t,
        method="BDF",
    )

    y_all = sol.y.T  # (Nt, 3 * Ny * Nx)
    u_all = y_all.reshape(Nt, Ny, Nx, 3)

    return u_all
