from pathlib import Path
import numpy as np

Darcy_2d_betta_01_dict = {
    'equation': 'div(a(x) * grad(u))',
    'right_part': 0.1,
    'definition_area': 'x in [0,1]^2',
    'boundary_condition': 'Dirichlet condition',
    'init_condition': 'random init condition for u(x) on the unit square',
    'inputs_var': (
        "x (np.ndarray): 1D array of x-coordinates, shape (Nx,)\n"
        "y (np.ndarray): 1D array of y-coordinates, shape (Ny,)\n"
        "a (np.ndarray): 2D coefficient array a(x,y), shape (Nx, Ny)"
    ),
    'outputs_var': 'u np.array with shape (Nx, Ny)'
}

ReacDiff_Nu05_Rho20_dict = {
    'equation': 'du/dt - 0.5 * d^2u/dx^2 - 2.0 * u * (1.0 - u)',
    'right_part': 0.0,
    'definition_area': 'x in [0,1], t in (0,1]',
    'boundary_condition': 'periodic',
    'init_condition': (
        'u(0, x) is a superposition of sinusoidal waves, '
        'normalized and with abs applied to avoid negative values'
    ),
    'inputs_var': (
        'x np.array with shape (Nx,), '
        't np.array with shape (Nt,), '
        'u(0, x) np.array with shape (Nx,)'
    ),
    'outputs_var': 'u np.array with shape (Nt, Nx)'
}

Burgers1D_Nu001_dict = {
    'equation': 'du/dt + d(0.5 * u**2)/dx - (0.01/np.pi) * d^2u/dx^2',
    'right_part': 0.0,
    'definition_area': 'x in (0,1), t in (0,2]',
    'boundary_condition': 'periodic',
    'init_condition': (
        'u(0, x) is a random superposition of sinusoidal waves '
        'u0(x) = sum_i A_i * sin(k_i * x + phi_i), with random amplitudes '
        'A_i in [0,1], integer modes k_i, random phases phi_i; '
        'absolute value and normalization are applied as in PDEBench Eq. (8)'
    ),
    'inputs_var': (
        'x np.array with shape (Nx,), '
        't np.array with shape (Nt,), '
        'u(0, x) np.array with shape (Nx,)'
    ),
    'outputs_var': 'u np.array with shape (Nt, Nx)'
}

Advection1D_Beta04_dict = {
    'equation': 'du/dt + 0.4 * du/dx',
    'right_part': 0.0,
    'definition_area': 'x in (0,1), t in (0,2]',
    'boundary_condition': 'periodic',
    'init_condition': (
        'u(0, x) is a random superposition of sinusoidal waves '
        'u0(x) = sum_i A_i * sin(k_i * x + phi_i) with k_max = 8 and N = 2; '
        'integer modes k_i chosen randomly, A_i in [0,1], phi_i in (0, 2*pi); '
        'absolute value and window function are applied with small probability'
    ),
    'inputs_var': (
        'x np.array with shape (Nx,), '
        't np.array with shape (Nt,), '
        'u(0, x) np.array with shape (Nx,)'
    ),
    'outputs_var': 'u np.array with shape (Nt, Nx)'
}

CompNS_2D_Eta1e2_Zeta1e2_M01_dict = {
    'equation': (
        'System in conservative/compressible form:\n'
        '1) d(rho)/dt + div(rho * v) = 0\n'
        '2) rho * (dv/dt + (v · grad)v) = -grad(p) + eta * Laplace(v) '
        '+ (zeta + eta/3) * grad(div(v))\n'
        '3) d/dt (eps + 0.5 * rho * |v|**2) + '
        'div(((eps + p + 0.5 * rho * |v|**2) * v - v · sigma_prime)) = 0\n'
        'with eps = p/(Gamma - 1), Gamma = 5/3'
    ),
    'right_part': 0.0,
    'definition_area': 'x in [0,1]^2, t in (0,2]',
    'boundary_condition': (
        'periodic in both spatial directions for all variables '
        '(rho, velocity components, pressure)'
    ),
    'init_condition': (
        'random field initial condition: density and pressure are given by a '
        'uniform background plus a perturbation constructed from a superposition '
        'of sinusoidal modes '
        'velocity v(x, t=0) is also a sum of sinusoidal waves with random '
        'amplitudes and phases, consistent with target Mach number M = 0.1; '
        'shear and bulk viscosities fixed to eta = 1e-2, zeta = 1e-2'
    ),
    'inputs_var': (
        'x np.array with shape (Nx,), '
        'y np.array with shape (Ny,), '
        't np.array with shape (Nt,), '
        'initial state at t=0 as np.array with shape (Nx, Ny, 4) '
        'representing [rho0, v0_x, v0_y, p0]'
    ),
    'outputs_var': (
        'state np.array with shape (Nt, Nx, Ny, 4), '
        'channels ordered as [rho, v_x, v_y, p]'
    )
}

IncompNS2D_InhomForcing_Nu001_dict = {
    'equation': (
        'Incompressible Navier–Stokes with forcing:\n'
        '1) div(v) = 0\n'
        '2) rho * (dv/dt + (v · grad)v) = -grad(p) + eta * Laplace(v) + u\n'
        'with constant density rho and viscosity nu = eta / rho'
    ),
    'right_part': 0.0,
    'definition_area': 'x in [0,1]^2, t in (0,1]',
    'boundary_condition': (
        'Dirichlet: velocity clamped to zero at the boundary '
        '(v = 0 on the perimeter of the unit square)'
    ),
    'init_condition': (
        'initial velocity v0(x,y) is drawn from an isotropic Gaussian random field '
        'with truncated power-law energy spectrum (tau_v = -3, sigma_v = 0.15); '
        'the forcing field u(x,y,t) is an inhomogeneous vector field, also drawn '
        'from a Gaussian random field with power-law spectrum (tau_u = -1, '
        'sigma_u = 0.4), fixed in time during each simulation; '
        'viscosity nu = 0.01, density rho taken constant'
    ),
    'inputs_var': (
        'x np.array with shape (Nx,), '
        'y np.array with shape (Ny,), '
        't np.array with shape (Nt,), '
        'v0(x,y) np.array with shape (Nx, Ny, 2) for initial velocity components, '
        'u(x,y) np.array with shape (Nx, Ny, 2) for forcing field'
    ),
    'outputs_var': (
        'v np.array with shape (Nt, Nx, Ny, 2) representing velocity field '
        'components [v_x, v_y]; '
        'pressure p can be reconstructed from the incompressibility constraint '
        'and Poisson equation if needed'
    )
}

ShallowWater2D_RadialDamBreak_dict = {
    'equation': (
        'System in conservative shallow-water form:\n'
        '1) dh/dt + d(h*u)/dx + d(h*v)/dy = 0\n'
        '2) d(h*u)/dt + d(u**2*h + 0.5 * g_r * h**2)/dx + d(u*v*h)/dy = - g_r * h * db/dx\n'
        '3) d(h*v)/dt + d(v**2*h + 0.5 * g_r * h**2)/dy + d(u*v*h)/dx = - g_r * h * db/dy'
    ),
    'right_part': 0.0,
    'definition_area': 'x, y in [-2.5, 2.5], t in (0, 1]',
    'boundary_condition': (
        'solid-wall (reflective) boundaries on all sides of the square domain, '
        'no normal flux across the boundary for the flow'
    ),
    'init_condition': (
        '2D radial dam-break scenario: water depth h(0, x, y) is a circular bump in the center\n'
        'h(0, x, y) = 2.0 for sqrt(x**2 + y**2) < r, and h(0, x, y) = 1.0 otherwise,\n'
        'where radius r is sampled from Uniform(0.3, 0.7). Initial horizontal velocities\n'
        'u(0, x, y) = 0, v(0, x, y) = 0. Bathymetry b(x, y) is a fixed spatial field, typically flat.'
    ),
    'inputs_var': (
        'x np.array with shape (Nx,), y np.array with shape (Ny,),\n'
        't np.array with shape (Nt,),\n'
        'b(x, y) np.array with shape (Nx, Ny) for bathymetry,\n'
        'h(0, x, y) np.array with shape (Nx, Ny),\n'
        'u(0, x, y) np.array with shape (Nx, Ny),\n'
        'v(0, x, y) np.array with shape (Nx, Ny)'
    ),
    'outputs_var': (
        'state np.array with shape (Nt, Nx, Ny, 3), channels ordered as [h, u, v]\n'
        'for water depth and horizontal velocity components'
    )
}

DiffSorp_1d_dict = {
    'equation': (
        '∂u/∂t = (D / R(u)) * ∂²u/∂x²,  '
        'R(u) = 1 + K_d * u          '
    ),
    'right_part': 0.0,
    'definition_area': 'x in [0,1], t in (0,1]',
    'boundary_condition': (
        'Neumann (no-flux): ∂u/∂x|_{x=0} = 0, ∂u/∂x|_{x=1} = 0'
    ),
    'init_condition': (
        'random non-negative concentration profile u(0,x): '
        'superposition of a few smooth bumps (Gaussians / sin waves), '
        'clipped to u ≥ 0'
    ),
    'inputs_var': (
        'x (np.ndarray): 1D array of spatial coordinates, shape (Nx,)\n'
        't (np.ndarray): 1D array of time points, shape (Nt,)\n'
        'u0 (np.ndarray): initial concentration u(0,x), shape (Nx,)'
    ),
    'outputs_var': (
        'u (np.ndarray): concentration field, shape (Nt, Nx)'
    )
}

DiffReact2D_FitzHughNagumo_dict = {
    'equation': (
        '2D diffusion-reaction system for activator u(t, x, y) and inhibitor v(t, x, y):\n'
        '  ∂u/∂t = D_u (∂²u/∂x² + ∂²u/∂y²) + R_u(u, v)\n'
        '  ∂v/∂t = D_v (∂²v/∂x² + ∂²v/∂y²) + R_v(u, v)\n'
        'Reaction terms (FitzHugh–Nagumo type):\n'
        '  R_u(u, v) = u - u**3 - k - v\n'
        '  R_v(u, v) = u - v\n'
        'with diffusion coefficients typically D_u = 1e-3, D_v = 5e-3 and k ≈ 5e-3.'
    ),
    'right_part': 0.0,
    'definition_area': 'x, y in [-1, 1], t in (0, 5]',
    'boundary_condition': (
        'homogeneous Neumann (no-flux) on all boundaries: \n'
        '  D_u * ∂u/∂n = 0,  D_v * ∂v/∂n = 0 on ∂Ω, where Ω = [-1, 1]²'
    ),
    'init_condition': (
        'Random noisy initial conditions for both activator and inhibitor on the 2D domain: \n'
        '  u(0, x, y) and v(0, x, y) are sampled as smooth random fields (e.g., Gaussian noise\n'
        '  filtered or normalized to a fixed amplitude range), independently for each sample.\n'
        'This matches the PDEBench setup where u and v start from random patterns that then evolve\n'
        'into structured spatio-temporal patterns due to diffusion and reaction dynamics.'
    ),
    'inputs_var': (
        'x (np.ndarray): 1D array of x-coordinates, shape (Nx,)\n'
        'y (np.ndarray): 1D array of y-coordinates, shape (Ny,)\n'
        't (np.ndarray): 1D array of time points, shape (Nt,)\n'
        'u0 (np.ndarray): initial activator field u(0, x, y), shape (Nx, Ny)\n'
        'v0 (np.ndarray): initial inhibitor field v(0, x, y), shape (Nx, Ny)'
    ),
    'outputs_var': (
        'state (np.ndarray): time evolution of activator and inhibitor fields,\n'
        '  shape (Nt, Nx, Ny, 2), with channels ordered as [u, v]'
    )
}

EQUATIONS_DATASET = {
    'Darcy_2d_betta_0.1': Darcy_2d_betta_01_dict,
    'ReacDiff_Nu05_Rho20': ReacDiff_Nu05_Rho20_dict,
    'Burgers1D_Nu001': Burgers1D_Nu001_dict,
    'Advection1D_Beta04': Advection1D_Beta04_dict,
    'CompNS_2D_Eta1e2_Zeta1e2_M01': CompNS_2D_Eta1e2_Zeta1e2_M01_dict,
    'IncompNS2D_InhomForcing_Nu001': IncompNS2D_InhomForcing_Nu001_dict,
    'ShallowWater2D_RadialDamBreak': ShallowWater2D_RadialDamBreak_dict,
    'DiffSorp_1d': DiffSorp_1d_dict,
    'DiffReact2D_FitzHughNagumo': DiffReact2D_FitzHughNagumo_dict

}

