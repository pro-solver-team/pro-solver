import numpy as np
from scipy.integrate import solve_ivp
from numba import njit


def solve_pde(x, t, u0, gamma=1.4, mu=0.01):
    """
    Solve a simplified 1D compressible Navier-Stokes system for (rho, u, p)
    with periodic boundary conditions.

    The PDE system (schematic form):

        continuity:        d rho / dt + d(rho * u) / dx = 0
        momentum:          d(rho u) / dt + d(rho u^2 + p) / dx = mu * d^2 u / dx^2
        pressure (energy): dp / dt + u * dp/dx + gamma * p * du/dx
                           = (gamma - 1) * mu * (du/dx)^2

    Parameters
    ----------
    x : np.array
        Spatial mesh with shape (Nx,)
    t : np.array
        Time mesh with shape (Nt,)
    u0 : np.array
        Initial condition with shape (Nx, 3),
        columns are [rho, u, p]
    gamma : float
        Ratio of specific heats
    mu : float
        Effective viscosity coefficient

    Returns
    -------
    u : np.array
        Solution with shape (Nt, Nx, 3)
        fields [rho, u, p]
    """
    Nx = len(x)
    Nt = len(t)

    if u0.ndim != 2 or u0.shape[0] != Nx or u0.shape[1] != 3:
        raise ValueError(
            f"u0 must have shape (Nx, 3), got {u0.shape}"
        )

    dx = x[1] - x[0]
    y0 = u0.reshape(-1)  # (3 * Nx,)

    @njit
    def grad_periodic(f):
        df = np.empty_like(f)
        df[1:-1] = (f[2:] - f[:-2]) / (2.0 * dx)
        df[0] = (f[1] - f[-1]) / (2.0 * dx)
        df[-1] = (f[0] - f[-2]) / (2.0 * dx)
        return df

    @njit
    def lap_periodic(f):
        lf = np.empty_like(f)
        lf[1:-1] = (f[2:] - 2.0 * f[1:-1] + f[:-2]) / (dx * dx)
        lf[0] = (f[1] - 2.0 * f[0] + f[-1]) / (dx * dx)
        lf[-1] = (f[0] - 2.0 * f[-1] + f[-2]) / (dx * dx)
        return lf

    @njit
    def pde_rhs(t_local, y):
        # unpack conservative and primitive variables
        rho = y[0:Nx]
        v = y[Nx:2 * Nx]
        p = y[2 * Nx:3 * Nx]

        drho_dx = grad_periodic(rho)
        dv_dx = grad_periodic(v)
        dp_dx = grad_periodic(p)
        dv_xx = lap_periodic(v)

        # continuity: rho_t + (rho v)_x = 0
        rhov = rho * v
        drhov_dx = grad_periodic(rhov)
        rho_t = -drhov_dx

        # momentum: (rho v)_t + (rho v^2 + p)_x = mu v_xx
        flux_m = rho * v * v + p
        dflux_m_dx = grad_periodic(flux_m)
        mom_t = -dflux_m_dx + mu * dv_xx

        # pressure equation (simplified energy balance)
        # p_t + v p_x + gamma p v_x = (gamma - 1) mu (v_x)^2
        p_t = -v * dp_dx - gamma * p * dv_dx + (gamma - 1.0) * mu * (dv_dx * dv_dx)

        dy = np.empty_like(y)
        dy[0:Nx] = rho_t
        dy[Nx:2 * Nx] = mom_t
        dy[2 * Nx:3 * Nx] = p_t
        return dy

    sol = solve_ivp(
        pde_rhs,
        (t[0], t[-1]),
        y0,
        t_eval=t,
        method="BDF",
    )

    y = sol.y.T  # (Nt, 3 * Nx)
    u = y.reshape(Nt, Nx, 3)
    return u
