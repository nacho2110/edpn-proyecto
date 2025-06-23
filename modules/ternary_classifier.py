from modules.binary_classifier import amp_polym, quantize
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

class TernaryClassifier:
    def __init__(self, x0: np.ndarray, sigma: float, s: int, r: int, rhor: float = 1/3, rhog: float = 1/3, rhob: float = 1/3):
        self.dim = len(x0.shape)
        self.grid = x0
        # param related to the quantization function
        self.sigma = sigma
        # param related to the amplification function
        self.rhor = rhor
        self.rhog = rhog
        self.rhob = rhob
        self.s = s
        # amplification functions for each color
        self.amp_r = lambda x: amp_polym(x, self.rhor)
        self.amp_g = lambda x: amp_polym(x, self.rhog)
        self.amp_b = lambda x: amp_polym(x, self.rhob)
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

    def _fr(self, x: float) -> float:
        #this goes to one if red is dominant in binary comparison
        return self.q_r(x + self.sigma * self.amp_r(x))

    def _fg(self, x: float) -> float:
        #this goes to one if green is dominant in binary comparison
        return self.q_g(x + self.sigma * self.amp_g(x))
    
    def _fb(self, x: float) -> float:
        #this goes to one if blue is dominant in binary comparison
        return self.q_b(x + self.sigma * self.amp_b(x))
    
    def alternated_step(self):
        new_grid = np.zeros(self.grid.shape)
        for x in np.ndindex(self.grid.shape[0:2]):
            neighbors = self.neighbors[x]
            
            neighbor_values_r = [self.grid[y][0] for y in neighbors]
            neighbor_values_g = [self.grid[y][1] for y in neighbors]
            neighbor_values_b = [self.grid[y][2] for y in neighbors]
            print(f"Processing pixel {x} with neighbors: {neighbors}")
            print(f"neighbor_values_r: {np.sum(neighbor_values_r)}")
            print(f"neighbor_values_g: {np.sum(neighbor_values_g)}")
            print(f"neighbor_values_b: {np.sum(neighbor_values_b)}")

            # SE PUEDE REFACTORIZAR ESTO Y DEJARLO COMO UNA FUNCION NMS PARA NO REPETIR TANTO CODIGO
            # TE LO ENCARGO PARA QUE TMBN PUEDAS CACHAR LO QUE HICE AQUI JSJSJS (me dio paja hacerlo ahora XDDD)
            if self.t % 3 == 0:
                # Update red channel vs green channel
                tot_red_val = np.sum(neighbor_values_r)
                tot_green_val = np.sum(neighbor_values_g)
                # test if there are values in red and green channels
                alpha = tot_green_val + tot_red_val 
                if alpha > 0:
                    # Calculate the average value for red channel 
                    avg_value_r = tot_red_val / alpha
                    print(f"avg_value_r: {avg_value_r}, alpha: {alpha}")
                    # the new value
                    new_value_r = self._fr(avg_value_r)
                    print(f"new_value_r: {new_value_r}")
                    # If the delta is positive, it means red is dominant
                    # in the other hand, if negative it means green is dominant
                    delta = new_value_r - avg_value_r
                    # we can update using + delta in red and -delta in green
                    new_value_r = (new_value_r + delta)*alpha
                    new_value_g = (1 - new_value_r)*alpha
                    new_grid[x] = [
                    new_value_r,  # Update red channel
                    new_value_g,  # Update green channel
                    self.grid[x][2]   # Keep blue channel unchanged
                    ]

                else: 
                    # If there are no values in red and green channels, keep them unchanged 
                    new_grid[x] = [
                        self.grid[x][0],  # Keep red channel unchanged
                        self.grid[x][1],  # Keep green channel unchanged
                        self.grid[x][2]   # Keep blue channel unchanged
                    ]

            elif self.t % 3 == 1:
                # Update green channel vs blue channel
                tot_green_val = np.sum(neighbor_values_g)
                tot_blue_val = np.sum(neighbor_values_b)
                # test if there are values in green and blue channels
                alpha = tot_green_val + tot_blue_val 
                if alpha > 0:
                    # Calculate the average value for green channel 
                    avg_value_g = tot_green_val / alpha
                    # the new value
                    new_value_g = self._fr(avg_value_g)
                    # If the delta is positive, it means green is dominant
                    # in the other hand, if negative it means blue is dominant
                    delta = new_value_g - avg_value_g
                    # we can update using + delta in green and -delta in blue
                    new_value_g = (new_value_g + delta)*alpha
                    new_value_b = (1 - new_value_g)*alpha
                    new_grid[x] = [
                    self.grid[x][0],  # Keep red channel unchanged
                    new_value_g,  # Update green channel
                    new_value_b  # Update blue channel
                    ]

                else: 
                    # If there are no values in green and blue channels, keep them unchanged 
                    new_grid[x] = [
                        self.grid[x][0],  # Keep red channel unchanged
                        self.grid[x][1],  # Keep green channel unchanged
                        self.grid[x][2]   # Keep blue channel unchanged
                    ]

            else:
                # Update blue channel vs red channel
                tot_blue_val = np.sum(neighbor_values_b)
                tot_red_val = np.sum(neighbor_values_r)
                # test if there are values in blue and red channels
                alpha = tot_blue_val + tot_red_val 
                if alpha > 0:
                    # Calculate the average value for blue channel 
                    avg_value_b = tot_blue_val / alpha
                    # the new value
                    new_value_b = self._fr(avg_value_b)
                    # If the delta is positive, it means blue is dominant
                    # in the other hand, if negative it means red is dominant
                    delta = new_value_b - avg_value_b
                    # we can update using + delta in blue and -delta in red
                    new_value_b = (new_value_b + delta)*alpha
                    new_value_r = (1 - new_value_b)*alpha
                    new_grid[x] = [
                        new_value_r,  # Update red channel
                        self.grid[x][1],  # Keep green channel unchanged
                        new_value_b   # Update blue channel
                    ]

                else: 
                    # If there are no values in blue and red channels, keep them unchanged 
                    new_grid[x] = [
                        self.grid[x][0],  # Keep red channel unchanged
                        self.grid[x][1],  # Keep green channel unchanged
                        self.grid[x][2]   # Keep blue channel unchanged
                    ]
        self.grid = new_grid
        self.history.append(np.mean(self.grid))
        self.t += 1

    # el show hay que ver como implementarlo, porque ahora es una imagen 3D
    # se me ocurre que se puede hacer un show por canal, o un show por color
    # incluso si lo podemos mapear con rbg estaria fino, pero no se como podría hacerse
    def show(self, ax=None, title=None):
        """
        Displays the current grid as an RGB image using matplotlib.
        """
        img = np.clip(self.grid, 0, 1)  # Ensure values are in [0, 1] for imshow
        if ax is None:
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            if title:
                plt.title(title)
            plt.axis('off')
            plt.show()
        else:
            ax.imshow(img)
            if title:
                ax.set_title(title)
            ax.axis('off')

