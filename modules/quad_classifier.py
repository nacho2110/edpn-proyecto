from modules.binary_classifier import amp_polym, quantize
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

class QuadClassifier:
    def __init__(self, x0: np.ndarray, sigma: float, s: int, r: int, rhor: float = 1/4, rhog: float = 1/4, rhob: float = 1/4, rho4= 1/4):
        self.dim = len(x0.shape)
        self.grid = x0
        # param related to the quantization function
        self.sigma = sigma
        # param related to the amplification function
        self.rhor = rhor
        self.rhog = rhog
        self.rhob = rhob
        self.rho4 = rho4
        self.s = s
        # amplification functions for each color
        self.amp_r = lambda x: amp_polym(x, self.rhor)
        self.amp_g = lambda x: amp_polym(x, self.rhog)
        self.amp_b = lambda x: amp_polym(x, self.rhob)
        self.amp_4 = lambda x: amp_polym(x, self.rho4)
        # time step
        self.t = 0
        # assertions to ensure valid parameters
        # assert related to rhor, rhog, rhob are omitted for simplicity
        assert sigma >= 0 and sigma <= 1
        assert sigma <= 2/3*s
        # params related to neighborhood 
        self.r = r
        # quantization functions
        self.q_r = lambda x: quantize(x, self.s, self.rhor)
        self.q_g = lambda x: quantize(x, self.s, self.rhog)
        self.q_b = lambda x: quantize(x, self.s, self.rhob)
        self.l_4 = lambda x: quantize(x, self.s, self.rho4)
        
        self.neighbors = {}  # Diccionario para almacenar vecinos
        grid_shape = x0.shape[0:2]  # Assuming x0 is a 3D array (height, width, channels)
        for x in np.ndindex(grid_shape):  # Cálculo de vecinos
            neighbors = [x]
            for offset in product(range(-self.r, self.r + 1), repeat=len(x)):
                if (np.sum([abs(o)for o in offset]) <= self.r) and any(o != 0 for o in offset):
                    y = tuple((xi + oi) % lim
                              for xi, oi, lim in zip(x, offset, grid_shape))
                    if y not in neighbors:
                        neighbors.append(y)
            self.neighbors[x] = neighbors
        self.history = [np.mean(x0)]  # Lista para almacenar el historial de la cuadrícula

    
