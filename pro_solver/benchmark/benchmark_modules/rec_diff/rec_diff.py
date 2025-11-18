import numpy as np
from scipy.linalg import solve_banded

def solve_pde(x, t, u0):
    # Parameters
    Nx = len(x)
    Nt = len(t)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    # Initialize solution array
    u = np.zeros((Nt, Nx))
    u[0, :] = u0
    
    # Precompute coefficients
    alpha = 0.5 * dt / dx**2
    beta = 2 * dt
    
    # Tridiagonal matrix coefficients
    a = np.ones(Nx) * (-alpha)
    b = np.ones(Nx) * (1 + 2 * alpha)
    c = np.ones(Nx) * (-alpha)
    
    # Periodic boundary conditions
    a[0] = 0
    c[-1] = 0
    b[0] += alpha
    b[-1] += alpha
    
    # Combine coefficients into a banded matrix
    ab = np.array([a, b, c])
    
    # Time stepping
    for n in range(1, Nt):
        # Nonlinear term
        nonlinear_term = beta * u[n-1, :] * (1 - u[n-1, :])
        
        # Right-hand side
        rhs = u[n-1, :] + nonlinear_term
        
        # Solve the tridiagonal system
        u[n, :] = solve_banded((1, 1), ab, rhs)
    
    return u