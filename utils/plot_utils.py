import torch
import numpy as np
import matplotlib.pyplot as plt

import time
import copy
import os

def plot_2d_surface(surface=None, scaler=1, xlim=(-1,1), ylim=(-1,1), num_levels=8):
    if not (type(surface).__module__ == np.__name__):
        surface = np.array(surface)
    x_grid, y_grid = np.meshgrid(np.unique(surface[:, 0]), np.unique(surface[:, 1]))

    loss_grid = surface[:, 2].reshape(x_grid.shape)

    loss_list = surface[:,2]
    levels_obj = ContourLevels(loss=loss_list, num_levels=num_levels)
    print(f"loss list:\n{loss_list}")
    levels = levels_obj.fit_levels()

    contourf = plt.contourf(x_grid, y_grid, loss_grid, levels=levels, cmap='Spectral', extend='both', zorder=0)
    plt.contour(x_grid, y_grid, loss_grid,  levels=levels, extend='both', cmap='Spectral', zorder=1)
    plt.colorbar(contourf)

    plt.xlim(xlim[0]*scaler, xlim[1]*scaler)
    plt.ylim(ylim[0]*scaler, ylim[1]*scaler)

    plt.title('2D Test Loss Surface in Iris Dataset')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f"{str(time.time())}.png")

def plot_3d_surface(surface=None, test=True, scaler=1):
    if not (type(surface).__module__ == np.__name__):
        surface = np.array(surface)
    x_grid, y_grid = np.meshgrid(np.unique(surface[:, 0]), np.unique(surface[:, 1]))
    if test:
        loss_grid = surface[:, 2].reshape(x_grid.shape)
    else:
        loss_grid = surface[:, 2].reshape(x_grid.shape)

    zlim=(0, 7)

    ax = plt.axes(projection='3d')
    colorbar = ax.plot_surface(x_grid, y_grid, loss_grid, vmin=loss_grid.min(), vmax=10, cmap='Spectral', zorder=1)
    plt.colorbar(colorbar)
    ax.set_title("3D Loss surface on Iris")
    ax.set_xlabel("X Alphas")
    ax.set_ylabel("Y Betas")
    ax.set_zlabel("Z Loss")
    ax.set_xlim(-20, 20)
    ax.set_ylim(-10, 20)
    ax.set_zlim(zlim[0], zlim[1])
    plt.savefig(f"{str(time.time())}_3d.png")
    plt.show()

class ContourLevels():
    def __init__(self, loss=None, num_levels=8, init=None):
        assert loss is not None
        self.loss = loss
        self.min_loss = np.min(loss)
        self.max_loss = np.max(loss)
        self.num_levels = num_levels
        self.init = init
        if init:
            self.level_limits = [self.init]
        else:
            self.level_limits = [self.min_loss]
        
    def normalize_loss(self):
        return ((self.loss - self.min_loss) / (self.max_loss - self.min_loss))
    
    def level_mapping(self, x):
        return np.power(x, 2)
    
    def fit_levels(self):
        for i in range(1, self.num_levels+1):
            x = i/(self.num_levels-1)
            self.level_limits.append((self.level_limits[-1] + self.level_mapping(x)))

        return np.array(self.level_limits)