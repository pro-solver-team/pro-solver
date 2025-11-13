import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def solve_pde(x, y, a):
    Nx, Ny = len(x), len(y)
    hx, hy = x[1] - x[0], y[1] - y[0]
    A = np.zeros((Nx * Ny, Nx * Ny))
    b = np.zeros(Nx * Ny)
    for i in range(Nx):
        for j in range(Ny):
            idx = i * Ny + j
            if i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1:
                A[idx, idx] = 1
                b[idx] = 0
            else:
                A[idx, idx] = -4 * a[i, j] / (hx * hy)
                if i > 0:
                    A[idx, idx - Ny] = a[i - 1, j] / (hx * hy)
                if i < Nx - 1:
                    A[idx, idx + Ny] = a[i + 1, j] / (hx * hy)
                if j > 0:
                    A[idx, idx - 1] = a[i, j - 1] / (hx * hy)
                if j < Ny - 1:
                    A[idx, idx + 1] = a[i, j + 1] / (hx * hy)
                b[idx] = 0.1 * (hx * hy)
    u = spsolve(A, b)
    u = u.reshape((Nx, Ny))
    return u