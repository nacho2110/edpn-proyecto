import numpy as np
from itertools import product


class BinaryClassifier:
    def __init__(self, x0: np.ndarray, sigma: float, amp: function[float, float], s: int, r: int, eps: float = 1e-3, rho: float = 1/2):
        self.eps = eps
        self.dim = len(x0.shape)
        self.grid = x0
        self.amp = amp
        self.sigma = sigma
        self.rho = rho
        assert sigma > 0 and sigma < 1
        assert sigma < 2/3*s
        assert s < 1/(rho) and s < 1/(1-rho)
        self.s = s
        self.r = r
        self.neighbors = {}  # Diccionario para almacenar vecinos
        grid_shape = x0.shape
        for x in np.ndindex(x0.shape):  # Cálculo de vecinos
            neighbors = []
            for offset in product(range(-self.r, self.r + 1), repeat=len(x)):
                if all(abs(o) <= self.r for o in offset):
                    y = tuple((xi + oi) % lim
                              if (xi+oi) >= 0 else lim-(xi+oi) % lim
                              for xi, oi, lim in zip(x, offset, grid_shape))
                    neighbors.append(y)
            self.neighbors[x] = neighbors
    
    def _f(self, x: float) -> float:
        return x + self.sigma*self.amp(x)

    # def transition(self):
    #     new_grid = np.copy(self.grid)
    #     for x in np.ndindex(self.grid.shape):
    #         neighbors = self.neighbors[x]
    #         neighbor_values = [self.grid[y] for y in neighbors]
    #         avg_value = np.mean(neighbor_values)
    #         pass
