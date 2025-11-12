import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def solve_pde(x, y, a):
    Nx, Ny = len(x), len(y)
    hx, hy = x[1] - x[0], y[1] - y[0]
    dx = 1 / hx
    dy = 1 / hy
    A = np.zeros((Nx * Ny, Nx * Ny))
    b = np.zeros(Nx * Ny)
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            idx = i * Ny + j
            A[idx, idx] = -2 * (a[i, j] / hx**2 + a[i, j] / hy**2)
            A[idx, idx - 1] = a[i, j] / hx**2
            A[idx, idx + 1] = a[i, j] / hx**2
            A[idx, idx - Ny] = a[i, j] / hy**2
            A[idx, idx + Ny] = a[i, j] / hy**2
            b[idx] = 0.1
    # Boundary conditions
    for j in range(Ny):
        A[j, j] = 1
        b[j] = 0
    for j in range(Ny):
        A[(Nx - 1) * Ny + j, (Nx - 1) * Ny + j] = 1
        b[(Nx - 1) * Ny + j] = 0
    for i in range(1, Nx - 1):
        A[i * Ny, i * Ny] = 1
        b[i * Ny] = 0
    for i in range(1, Nx - 1):
        A[(i + 1) * Ny - 1, (i + 1) * Ny - 1] = 1
        b[(i + 1) * Ny - 1] = 0
    u = spsolve(A, b)
    return u.reshape((Nx, Ny))