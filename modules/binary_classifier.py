import numpy as np
import matplotlib.pyplot as plt
from itertools import product


def amp_polym(x: float, rc: float) -> float:
    """Función de amplificación cúbica."""
    return x*(1-x)*(x-rc)


def quantize(x: float, s: int, rc: float) -> float:
    """Cuantiza el valor x a un nivel de cuantización s."""
    if x >= rc:
        return np.max([0, np.ceil(s *(x-rc))/s + rc, 0])
    else:
        return np.min([np.floor(s *(x-rc))/s + rc, 1])


class BinaryClassifier:
    def __init__(self, x0: np.ndarray, sigma: float, s: int, r: int, rho: float = 1/2):
        self.dim = len(x0.shape)
        self.grid = x0
        self.sigma = sigma
        self.rho = rho
        self.amp = lambda x: amp_polym(x, self.rho)
        self.t = 0
        assert sigma >= 0 and sigma <= 1
        assert sigma <= 2/3*s
        assert sigma <= 1/(rho) and sigma <= 1/(1-rho)
        self.s = s
        self.r = r
        self.q = lambda x: quantize(x, self.s, self.rho)
        self.neighbors = {}  # Diccionario para almacenar vecinos
        grid_shape = x0.shape
        for x in np.ndindex(x0.shape):  # Cálculo de vecinos
            neighbors = [x]
            for offset in product(range(-self.r, self.r + 1), repeat=len(x)):
                if (np.sum([abs(o)for o in offset]) <= self.r) and any(o != 0 for o in offset):
                    y = tuple((xi + oi) % lim
                              for xi, oi, lim in zip(x, offset, grid_shape))
                    if y not in neighbors:
                        neighbors.append(y)
            self.neighbors[x] = neighbors
        self.history = [np.mean(x0)]  # Lista para almacenar el historial de la cuadrícula

    def _f(self, x: float) -> float:
        return self.q(x + self.sigma*self.amp(x))

    def step(self):
        new_grid = np.zeros(self.grid.shape)
        for x in np.ndindex(self.grid.shape):
            neighbors = self.neighbors[x]
            neighbor_values = [self.grid[y] for y in neighbors]
            avg_value = np.mean(neighbor_values)
            new_grid[x] = self._f(avg_value)
        self.grid = new_grid
        self.history.append(np.mean(self.grid))
        self.t += 1

    def show(self):
        plt.imshow(self.grid, cmap='binary', interpolation='nearest')
        plt.clim(0,1)
        plt.colorbar()
        plt.title(f'Clasificador binario (s={self.s}, r={self.r},'+\
                  f'σ={self.sigma}, t={self.t})')
