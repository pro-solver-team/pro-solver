import matplotlib.pyplot as plt
import imageio
import numpy as np
import os

def plot_evolution(u: np.ndarray, path="evolution.png"):
    plt.figure(figsize=(8, 5))
    plt.imshow(u.T, aspect="auto", origin="lower")
    plt.xlabel("time step")
    plt.ylabel("x coordinate")
    plt.colorbar(label="u(t, x)")
    plt.title("Solution evolution")
    plt.savefig(path)
    print(f"Saved: {path}")