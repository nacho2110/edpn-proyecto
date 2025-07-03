from modules.binary_classifier import amp_polym, quantize
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

class QuadClassifier:
    def __init__(self, x0: np.ndarray, sigma: float, s: int, r: int, rhor: float = 1/4, rhog: float = 1/4, rhob: float = 1/4, rho4= 1/4, mode= 0):
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
        self.q_4 = lambda x: quantize(x, self.s, self.rho4)
        
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
    
    def _f4(self, x: float) -> float:
        #this goes to one if 4th color is dominant in binary comparison
        return self.q_4(x + self.sigma * self.amp_4(x))
    
    def alternated_step_RGBY(self):
        new_grid = np.zeros(self.grid.shape)
        for x in np.ndindex(self.grid.shape[0:2]):
            neighbors = self.neighbors[x]
            
            neighbor_values_r = [self.grid[y][0] for y in neighbors]
            neighbor_values_g = [self.grid[y][1] for y in neighbors]
            neighbor_values_b = [self.grid[y][2] for y in neighbors]
            neighbor_values_4 = [self.grid[y][3] for y in neighbors]
            print(f"Processing pixel {x} with neighbors: {neighbors}")
            print(f"neighbor_values_r: {np.sum(neighbor_values_r)}")
            print(f"neighbor_values_g: {np.sum(neighbor_values_g)}")
            print(f"neighbor_values_b: {np.sum(neighbor_values_b)}")
            print(f"neighbor_values_4: {np.sum(neighbor_values_4)}")

            # SE PUEDE REFACTORIZAR ESTO Y DEJARLO COMO UNA FUNCION NMS PARA NO REPETIR TANTO CODIGO
            # TE LO ENCARGO PARA QUE TMBN PUEDAS CACHAR LO QUE HICE AQUI JSJSJS (me dio paja hacerlo ahora XDDD)

            if self.t % 4 == 0:
                # Update red channel vs green channel
                tot_red_val = np.sum(neighbor_values_r)
                tot_green_val = np.sum(neighbor_values_g)
                # test if there are values in red and green channels
                alpha = tot_green_val + tot_red_val 
                if alpha > 0:
                    # Calculate the average value for red channel 
                    avg_value_r = tot_red_val / alpha
                    # the new value
                    new_value_r = self._fr(avg_value_r)
                    # If the delta is positive, it means red is dominant
                    # in the other hand, if negative it means green is dominant
                    delta = new_value_r - avg_value_r
                    # we can update using + delta in red and -delta in green
                    final_r = np.clip((new_value_r + delta)*(1-self.grid[x][2] - self.grid[x][3]), 0, 1)
                    final_g = np.clip((1 - new_value_r)*(1-self.grid[x][2] - self.grid[x][3]), 0, 1)
            
                    new_grid[x] = [
                    final_r,  # Update red channel
                    final_g,  # Update green channel
                    self.grid[x][2],   # Keep blue channel unchanged
                    self.grid[x][3]   # Keep 4th channel unchanged
                    ]

                else: 
                    # If there are no values in red and green channels, keep them unchanged 
                    new_grid[x] = [
                        self.grid[x][0],  # Keep red channel unchanged
                        self.grid[x][1],  # Keep green channel unchanged
                        self.grid[x][2],  # Keep blue channel unchanged
                        self.grid[x][3]   # Keep 4th channel unchanged
                    ]

            elif self.t % 4 == 1:
                # Update green channel vs blue channel
                tot_green_val = np.sum(neighbor_values_g)
                tot_blue_val = np.sum(neighbor_values_b)
                # test if there are values in green and blue channels
                alpha = tot_green_val + tot_blue_val 
                if alpha > 0:
                    # Calculate the average value for green channel 
                    avg_value_g = tot_green_val / alpha
                    # the new value
                    new_value_g = self._fg(avg_value_g)
                    # If the delta is positive, it means green is dominant
                    # in the other hand, if negative it means blue is dominant
                    delta = new_value_g - avg_value_g
                    # we can update using + delta in green and -delta in blue
                    final_g = np.clip((new_value_g + delta)*(1 - self.grid[x][0] - self.grid[x][3]), 0, 1)
                    final_b = np.clip((1 - new_value_g)*(1 - self.grid[x][0] - self.grid[x][3]), 0, 1)
                    new_grid[x] = [
                    self.grid[x][0],  # Keep red channel unchanged
                    final_g,  # Update green channel
                    final_b,  # Update blue channel
                    self.grid[x][3]   # Keep 4th channel unchanged
                    ]

                else: 
                    # If there are no values in green and blue channels, keep them unchanged 
                    new_grid[x] = [
                        self.grid[x][0],  # Keep red channel unchanged
                        self.grid[x][1],  # Keep green channel unchanged
                        self.grid[x][2],  # Keep blue channel unchanged
                        self.grid[x][3]   # Keep 4th channel unchanged
                    ]

            elif self.t % 4 == 2:
                # Update blue channel vs 4th channel
                tot_blue_val = np.sum(neighbor_values_b)
                tot_4_val = np.sum(neighbor_values_4)
                # test if there are values in blue and 4th channels
                alpha = tot_blue_val + tot_4_val 
                if alpha > 0:
                    # Calculate the average value for blue channel 
                    avg_value_b = tot_blue_val / alpha
                    # the new value
                    new_value_b = self._fb(avg_value_b)
                    # If the delta is positive, it means blue is dominant
                    # in the other hand, if negative it means 4th color is dominant
                    delta = new_value_b - avg_value_b
                    # we can update using + delta in blue and -delta in 4th color
                    final_b = np.clip((new_value_b + delta)*(1 - self.grid[x][0] - self.grid[x][1]), 0, 1)
                    final_4 = np.clip((1 - new_value_b)*(1 - self.grid[x][0] - self.grid[x][1]), 0, 1)
                    new_grid[x] = [
                        self.grid[x][0],  # Keep red channel unchanged
                        self.grid[x][1],  # Keep green channel unchanged
                        final_b,  # Update blue channel
                        final_4   # Update 4th channel
                    ]

                else: 
                    # If there are no values in blue and 4th channels, keep them unchanged 
                    new_grid[x] = [
                        self.grid[x][0],  # Keep red channel unchanged
                        self.grid[x][1],  # Keep green channel unchanged
                        self.grid[x][2],  # Keep blue channel unchanged
                        self.grid[x][3]   # Keep 4th channel unchanged
                    ]

            else:
                # Update 4th channel vs red channel
                tot_4_val = np.sum(neighbor_values_4)
                tot_red_val = np.sum(neighbor_values_r)
                # test if there are values in 4th and red channels
                alpha = tot_4_val + tot_red_val 
                if alpha > 0:
                    # Calculate the average value for 4th channel 
                    avg_value_4 = tot_4_val / alpha
                    # the new value
                    new_value_4 = self._f4(avg_value_4)
                    # If the delta is positive, it means 4th color is dominant
                    # in the other hand, if negative it means red is dominant
                    delta = new_value_4 - avg_value_4
                    # we can update using + delta in 4th color and -delta in red
                    final_4 = np.clip((new_value_4 + delta)*(1 - self.grid[x][1] - self.grid[x][2]), 0, 1)
                    final_r = np.clip((1 - new_value_4)*(1 - self.grid[x][1] - self.grid[x][2]), 0, 1)
                    new_grid[x] = [
                        final_r,  # Update red channel
                        self.grid[x][1],  # Keep green channel unchanged
                        self.grid[x][2],   # Keep blue channel unchanged
                        final_4   # Update 4th channel
                    ]

                else: 
                    # If there are no values in 4th and red channels, keep them unchanged 
                    new_grid[x] = [
                        self.grid[x][0],  # Keep red channel unchanged
                        self.grid[x][1],  # Keep green channel unchanged
                        self.grid[x][2],  # Keep blue channel unchanged
                        self.grid[x][3]   # Keep 4th channel unchanged
                    ]

        # Update the grid with the new values
        self.grid = new_grid
        self.history.append(np.mean(self.grid))

        # PARA RANDOMIZAR SE PUEDE USAR 
        self.t += np.random.randint(1, 5)  # Randomly increment t by 1 to 4
        # self.t += 1



    def alternated_step_RBYG(self):
        new_grid = np.zeros(self.grid.shape)
        for x in np.ndindex(self.grid.shape[0:2]):
            neighbors = self.neighbors[x]
            
            neighbor_values_r = [self.grid[y][0] for y in neighbors]
            neighbor_values_g = [self.grid[y][1] for y in neighbors]
            neighbor_values_b = [self.grid[y][2] for y in neighbors]
            neighbor_values_4 = [self.grid[y][3] for y in neighbors]
            print(f"Processing pixel {x} with neighbors: {neighbors}")
            print(f"neighbor_values_r: {np.sum(neighbor_values_r)}")
            print(f"neighbor_values_g: {np.sum(neighbor_values_g)}")
            print(f"neighbor_values_b: {np.sum(neighbor_values_b)}")
            print(f"neighbor_values_4: {np.sum(neighbor_values_4)}")

            if self.t % 4 == 0:
                # Update red channel vs blue channel
                tot_red_val = np.sum(neighbor_values_r)
                tot_blue_val = np.sum(neighbor_values_b)
                # test if there are values in red and blue channels
                alpha = tot_red_val + tot_blue_val 
                if alpha > 0:
                    # Calculate the average value for red channel 
                    avg_value_r = tot_red_val / alpha
                    # the new value
                    new_value_r = self._fr(avg_value_r)
                    # If the delta is positive, it means red is dominant
                    # in the other hand, if negative it means blue is dominant
                    delta = new_value_r - avg_value_r
                    # we can update using + delta in red and -delta in blue
                    final_r = np.clip((new_value_r + delta)*(1-self.grid[x][1] - self.grid[x][3]), 0, 1)
                    final_b = np.clip((1 - new_value_r)*(1-self.grid[x][1] - self.grid[x][3]), 0, 1)
            
                    new_grid[x] = [
                    final_r,  # Update red channel
                    self.grid[x][1],  # Keep green channel unchanged
                    final_b,  # Update blue channel
                    self.grid[x][3]   # Keep 4th channel unchanged
                    ]

                else: 
                    # If there are no values in red and blue channels, keep them unchanged 
                    new_grid[x] = [
                        self.grid[x][0],  # Keep red channel unchanged
                        self.grid[x][1],  # Keep green channel unchanged
                        self.grid[x][2],  # Keep blue channel unchanged
                        self.grid[x][3]   # Keep 4th channel unchanged
                    ]
            elif self.t % 4 == 1:
                # Update blue channel vs yellow channel
                tot_blue_val = np.sum(neighbor_values_b)
                tot_yellow_val = np.sum(neighbor_values_4)  # Assuming the 4th channel is yellow
                # test if there are values in blue and yellow channels
                alpha = tot_blue_val + tot_yellow_val 
                if alpha > 0:
                    # Calculate the average value for blue channel 
                    avg_value_b = tot_blue_val / alpha
                    # the new value
                    new_value_b = self._fb(avg_value_b)
                    # If the delta is positive, it means blue is dominant
                    # in the other hand, if negative it means yellow is dominant
                    delta = new_value_b - avg_value_b
                    # we can update using + delta in blue and -delta in yellow
                    final_b = np.clip((new_value_b + delta)*(1 - self.grid[x][0] - self.grid[x][1]), 0, 1)
                    final_y = np.clip((1 - new_value_b)*(1 - self.grid[x][0] - self.grid[x][1]), 0, 1)
                    new_grid[x] = [
                        self.grid[x][0],  # Keep red channel unchanged
                        self.grid[x][1],  # Keep green channel unchanged
                        final_b,  # Update blue channel
                        final_y   # Update yellow channel
                    ]

                else: 
                    # If there are no values in blue and yellow channels, keep them unchanged 
                    new_grid[x] = [
                        self.grid[x][0],  # Keep red channel unchanged
                        self.grid[x][1],  # Keep green channel unchanged
                        self.grid[x][2],  # Keep blue channel unchanged
                        self.grid[x][3]   # Keep yellow channel unchanged
                    ]
            elif self.t % 4 == 2:
                # Update yellow channel vs green channel
                tot_yellow_val = np.sum(neighbor_values_4)
                tot_green_val = np.sum(neighbor_values_g)
                # test if there are values in yellow and green channels
                alpha = tot_yellow_val + tot_green_val 
                if alpha > 0:
                    # Calculate the average value for yellow channel 
                    avg_value_y = tot_yellow_val / alpha
                    # the new value
                    new_value_y = self._f4(avg_value_y)
                    # If the delta is positive, it means yellow is dominant
                    # in the other hand, if negative it means green is dominant
                    delta = new_value_y - avg_value_y
                    # we can update using + delta in yellow and -delta in green
                    final_y = np.clip((new_value_y + delta)*(1 - self.grid[x][0] - self.grid[x][2]), 0, 1)
                    final_g = np.clip((1 - new_value_y)*(1 - self.grid[x][0] - self.grid[x][2]), 0, 1)
                    new_grid[x] = [
                        self.grid[x][0],  # Keep red channel unchanged
                        final_g,  # Update green channel
                        self.grid[x][2],   # Keep blue channel unchanged
                        final_y   # Update yellow channel
                    ]

                else: 
                    # If there are no values in yellow and green channels, keep them unchanged 
                    new_grid[x] = [
                        self.grid[x][0],  # Keep red channel unchanged
                        self.grid[x][1],  # Keep green channel unchanged
                        self.grid[x][2],  # Keep blue channel unchanged
                        self.grid[x][3]   # Keep yellow channel unchanged
                    ]
            else:
                # Update green channel vs red channel
                tot_green_val = np.sum(neighbor_values_g)
                tot_red_val = np.sum(neighbor_values_r)
                # test if there are values in green and red channels
                alpha = tot_green_val + tot_red_val 
                if alpha > 0:
                    # Calculate the average value for green channel 
                    avg_value_g = tot_green_val / alpha
                    # the new value
                    new_value_g = self._fg(avg_value_g)
                    # If the delta is positive, it means green is dominant
                    # in the other hand, if negative it means red is dominant
                    delta = new_value_g - avg_value_g
                    # we can update using + delta in green and -delta in red
                    final_g = np.clip((new_value_g + delta)*(1 - self.grid[x][2] - self.grid[x][3]), 0, 1)
                    final_r = np.clip((1 - new_value_g)*(1 - self.grid[x][2] - self.grid[x][3]), 0, 1)
                    new_grid[x] = [
                        final_r,  # Update red channel
                        final_g,  # Update green channel
                        self.grid[x][2],   # Keep blue channel unchanged
                        self.grid[x][3]   # Keep yellow channel unchanged
                    ]

                else: 
                    # If there are no values in green and red channels, keep them unchanged 
                    new_grid[x] = [
                        self.grid[x][0],  # Keep red channel unchanged
                        self.grid[x][1],  # Keep green channel unchanged
                        self.grid[x][2],  # Keep blue channel unchanged
                        self.grid[x][3]   # Keep yellow channel unchanged
                    ]
                

    def step(self):
        if self.mode == 0:
            self.alternated_step_RGBY()
        elif self.mode == 1:
            self.alternated_step_RBYG()
    


    def show(self, ax=None, title=None):
        # Pasa los valores de la grilla a valores RGB, esto es, R, G y B se mantienen igual
        # pero el 4to color amarillo (Y), que es [0, 0, 0 , 1], se transforma a [1, 222/255, 0] en RGB
        """
        Displays the current grid as an RGB image using matplotlib.
        """
        m = self.grid.shape[0]
        n = self.grid.shape[1]
        img = np.zeros((m, n, 3))
        for i in range(m):
            for j in range(n):
                x = self.grid[i, j, :]
                e1= np.array([1, 0, 0])
                e2= np.array([0, 1, 0])
                e3= np.array([0, 0, 1])
                e4= np.array([1, 222/255, 0])
                img[i,j] = np.clip(x[0] * e1 + x[1] * e2 + x[2] * e3 + x[3] * e4 ,0, 1)
                
        
        img = (img.reshape(img.shape[0], img.shape[1], 3) * 255).astype(np.uint8)# Ensure values are in [0, 1] for imshow
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



    

    
