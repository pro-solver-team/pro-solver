import argparse
from pathlib import Path

import h5py
import numpy as np

from pro_solver.benchmark.data_modules.data_vars import (
    # уже существующие
    reacdiff1d_path,
    darcy2d_path,  # не трогаем, просто импорт, если понадобится позже

    # новые 1D
    burgers1d_path,
    advection1d_path,
    diffsorp1d_path,
    compressible_ns1d_path,

    # новые 2D
    incomp_ns2d_path,
    shallow_water2d_path,
    reacdiff2d_path,
)

# солверы, которые мы уже создали в benchmark_modules/*
from pro_solver.benchmark.benchmark_modules.burgers1d import solve_pde as solve_burgers1d
from pro_solver.benchmark.benchmark_modules.advection1d import solve_pde as solve_advection1d
from pro_solver.benchmark.benchmark_modules.diff_sorp1d import solve_pde as solve_diffsorp1d
from pro_solver.benchmark.benchmark_modules.compressible_ns1d import solve_pde as solve_compns1d
from pro_solver.benchmark.benchmark_modules.incomp_ns2d import solve_pde as solve_incompns2d
from pro_solver.benchmark.benchmark_modules.shallow_water2d import solve_pde as solve_shallow2d
from pro_solver.benchmark.benchmark_modules.rec_diff2d import solve_pde as solve_recdiff2d


# ---------- helpers ----------

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------- 1D SCALAR EQUATIONS: Burgers, Advection, Diffusion–Sorption ----------

def generate_1d_scalar_dataset(
    out_path: Path,
    solver_fn,
    num_samples: int = 64,
    nx: int = 256,
    nt: int = 200,
    x_start: float = 0.0,
    x_end: float = 1.0,
    t_final: float = 2.0,
    solver_kwargs=None,
) -> None:
    """
    Общий генератор для 1D задач с одним скалярным полем u(t, x):
    Burgers, Advection, Diffusion-Sorption.
    """
    if solver_kwargs is None:
        solver_kwargs = {}

    _ensure_parent(out_path)

    x = np.linspace(x_start, x_end, nx, endpoint=False)
    t = np.linspace(0.0, t_final, nt)

    tensor = np.zeros((num_samples, nt, nx), dtype=np.float32)

    for i in range(num_samples):
        # разные типы начальных условий для разнообразия
        if i % 3 == 0:
            u0 = np.sin(2.0 * np.pi * x)
        elif i % 3 == 1:
            u0 = np.exp(-((x - 0.5) ** 2) / 0.01)
        else:
            u0 = np.sin(4.0 * np.pi * x) * np.exp(-((x - 0.3) ** 2) / 0.02)

        u = solver_fn(x, t, u0.astype(np.float64), **solver_kwargs)  # (Nt, Nx)
        tensor[i] = u.astype(np.float32)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("t-coordinate", data=t)
        f.create_dataset("x-coordinate", data=x)
        f.create_dataset("tensor", data=tensor)

    print(f"[generate] saved 1D scalar dataset to {out_path}")
    print(f"  tensor shape: {tensor.shape}")


def generate_burgers1d(num_samples: int = 64) -> None:
    generate_1d_scalar_dataset(
        out_path=burgers1d_path,
        solver_fn=solve_burgers1d,
        num_samples=num_samples,
        nx=256,
        nt=200,
        t_final=2.0,
        solver_kwargs={"nu": 0.1},
    )


def generate_advection1d(num_samples: int = 64) -> None:
    generate_1d_scalar_dataset(
        out_path=advection1d_path,
        solver_fn=solve_advection1d,
        num_samples=num_samples,
        nx=256,
        nt=200,
        t_final=2.0,
        solver_kwargs={"c": 1.0},
    )


def generate_diffsorp1d(num_samples: int = 64) -> None:
    generate_1d_scalar_dataset(
        out_path=diffsorp1d_path,
        solver_fn=solve_diffsorp1d,
        num_samples=num_samples,
        nx=256,
        nt=200,
        t_final=2.0,
        solver_kwargs={"D": 0.5, "k": 1.0},
    )


# ---------- 1D COMPRESSIBLE NAVIER–STOKES (3 поля: rho, u, p) ----------

def generate_compressible_ns1d(
    num_samples: int = 16,
    nx: int = 128,
    nt: int = 150,
    x_start: float = 0.0,
    x_end: float = 1.0,
    t_final: float = 1.0,
) -> None:
    """
    Генерация датасета для 1D Compressible NS:
    tensor[i, j, k, :] = [rho, u, p]
    """
    out_path = compressible_ns1d_path
    _ensure_parent(out_path)

    x = np.linspace(x_start, x_end, nx, endpoint=False)
    t = np.linspace(0.0, t_final, nt)

    tensor = np.zeros((num_samples, nt, nx, 3), dtype=np.float32)

    for i in range(num_samples):
        # rho >= 0, небольшие вариации вокруг 1
        rho0 = 1.0 + 0.2 * np.sin(2.0 * np.pi * x * (i + 1) / (num_samples + 1))
        # скорость
        u0 = 0.5 * np.sin(2.0 * np.pi * x)
        # давление
        p0 = 1.0 + 0.1 * np.cos(2.0 * np.pi * x)

        u0_field = np.stack([rho0, u0, p0], axis=-1)  # (Nx, 3)

        u = solve_compns1d(x, t, u0_field.astype(np.float64))  # (Nt, Nx, 3)
        tensor[i] = u.astype(np.float32)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("t-coordinate", data=t)
        f.create_dataset("x-coordinate", data=x)
        f.create_dataset("tensor", data=tensor)

    print(f"[generate] saved 1D compressible NS dataset to {out_path}")
    print(f"  tensor shape: {tensor.shape}")


# ---------- 2D SCALAR: 2D Diffusion–Reaction ----------

def generate_recdiff2d(
    num_samples: int = 16,
    nx: int = 64,
    ny: int = 64,
    nt: int = 100,
    x_start: float = 0.0,
    x_end: float = 1.0,
    y_start: float = 0.0,
    y_end: float = 1.0,
    t_final: float = 1.0,
) -> None:
    """
    2D diffusion–reaction: tensor[i, j, ky, kx] = u(t_j, y_ky, x_kx)
    """
    out_path = reacdiff2d_path
    _ensure_parent(out_path)

    x = np.linspace(x_start, x_end, nx, endpoint=False)
    y = np.linspace(y_start, y_end, ny, endpoint=False)
    t = np.linspace(0.0, t_final, nt)

    tensor = np.zeros((num_samples, nt, ny, nx), dtype=np.float32)

    X, Y = np.meshgrid(x, y)

    for i in range(num_samples):
        if i % 2 == 0:
            u0 = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)
        else:
            u0 = np.sin(2.0 * np.pi * X) * np.sin(2.0 * np.pi * Y)

        u = solve_recdiff2d(x, y, t, u0.astype(np.float64))  # (Nt, Ny, Nx)
        tensor[i] = u.astype(np.float32)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("t-coordinate", data=t)
        f.create_dataset("x-coordinate", data=x)
        f.create_dataset("y-coordinate", data=y)
        f.create_dataset("tensor", data=tensor)

    print(f"[generate] saved 2D rec-diff dataset to {out_path}")
    print(f"  tensor shape: {tensor.shape}")


# ---------- 2D VECTOR: Incompressible NS (u, v) ----------

def generate_incomp_ns2d(
    num_samples: int = 8,
    nx: int = 64,
    ny: int = 64,
    nt: int = 80,
    x_start: float = 0.0,
    x_end: float = 1.0,
    y_start: float = 0.0,
    y_end: float = 1.0,
    t_final: float = 0.5,
) -> None:
    """
    2D inhomogeneous incompressible NS-like:
    tensor[i, j, ky, kx, :] = [u, v]
    """
    out_path = incomp_ns2d_path
    _ensure_parent(out_path)

    x = np.linspace(x_start, x_end, nx, endpoint=False)
    y = np.linspace(y_start, y_end, ny, endpoint=False)
    t = np.linspace(0.0, t_final, nt)

    tensor = np.zeros((num_samples, nt, ny, nx, 2), dtype=np.float32)

    X, Y = np.meshgrid(x, y)

    for i in range(num_samples):
        u0 = np.sin(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Y)
        v0 = -np.cos(2.0 * np.pi * X) * np.sin(2.0 * np.pi * Y)
        u0_field = np.stack([u0, v0], axis=-1)  # (Ny, Nx, 2)

        u = solve_incompns2d(x, y, t, u0_field.astype(np.float64))
        tensor[i] = u.astype(np.float32)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("t-coordinate", data=t)
        f.create_dataset("x-coordinate", data=x)
        f.create_dataset("y-coordinate", data=y)
        f.create_dataset("tensor", data=tensor)

    print(f"[generate] saved 2D incompressible NS dataset to {out_path}")
    print(f"  tensor shape: {tensor.shape}")


# ---------- 2D VECTOR: Shallow-Water (h, u, v) ----------

def generate_shallow_water2d(
    num_samples: int = 8,
    nx: int = 64,
    ny: int = 64,
    nt: int = 80,
    x_start: float = 0.0,
    x_end: float = 1.0,
    y_start: float = 0.0,
    y_end: float = 1.0,
    t_final: float = 0.5,
) -> None:
    """
    2D shallow-water:
    tensor[i, j, ky, kx, :] = [h, u, v]
    """
    out_path = shallow_water2d_path
    _ensure_parent(out_path)

    x = np.linspace(x_start, x_end, nx, endpoint=False)
    y = np.linspace(y_start, y_end, ny, endpoint=False)
    t = np.linspace(0.0, t_final, nt)

    tensor = np.zeros((num_samples, nt, ny, nx, 3), dtype=np.float32)

    X, Y = np.meshgrid(x, y)

    for i in range(num_samples):
        # высота воды: базовый уровень + "горб"
        h0 = 1.0 + 0.2 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)
        # небольшие вихревые скорости
        u0 = - (Y - 0.5)
        v0 = (X - 0.5)

        u0_field = np.stack([h0, u0, v0], axis=-1)  # (Ny, Nx, 3)

        u = solve_shallow2d(x, y, t, u0_field.astype(np.float64))
        tensor[i] = u.astype(np.float32)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("t-coordinate", data=t)
        f.create_dataset("x-coordinate", data=x)
        f.create_dataset("y-coordinate", data=y)
        f.create_dataset("tensor", data=tensor)

    print(f"[generate] saved 2D shallow-water dataset to {out_path}")
    print(f"  tensor shape: {tensor.shape}")


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PDE datasets for pro-solver benchmarks."
    )
    parser.add_argument(
        "--equation",
        type=str,
        required=True,
        choices=[
            "burgers1d",
            "advection1d",
            "diff_sorp1d",
            "compressible_ns1d",
            "reacdiff2d",
            "incomp_ns2d",
            "shallow_water2d",
        ],
        help="Имя уравнения, для которого генерируем датасет.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=32,
        help="Количество примеров (траекторий) в датасете.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    eq = args.equation
    n = args.num_samples

    if eq == "burgers1d":
        generate_burgers1d(num_samples=n)
    elif eq == "advection1d":
        generate_advection1d(num_samples=n)
    elif eq == "diff_sorp1d":
        generate_diffsorp1d(num_samples=n)
    elif eq == "compressible_ns1d":
        generate_compressible_ns1d(num_samples=n)
    elif eq == "reacdiff2d":
        generate_recdiff2d(num_samples=n)
    elif eq == "incomp_ns2d":
        generate_incomp_ns2d(num_samples=n)
    elif eq == "shallow_water2d":
        generate_shallow_water2d(num_samples=n)
    else:
        raise ValueError(f"Unknown equation: {eq}")


if __name__ == "__main__":
    main()
