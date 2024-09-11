import torch
import numpy as np

from pydive.types import Number


class Plane():
    def __init__(self, xlim: Number=(-1.0, 1.0), ylim: Number=(-1.0, 1.0), grid_size: int=51):
        self._xlim = xlim
        self._ylim = ylim
        self._grid_size = grid_size
        
        self.xlinspace = None
        self.ylinspace = None

    
    def __limit(self, xlim: tuple=None, ylim: tuple=None):
        if xlim and ylim:
            self.xlim = xlim
            self.ylim = ylim
    

    def xlimit(self, xlim: tuple=None) -> None:
        if xlim:
            self.xlim = xlim


    def ylimit(self, ylim: tuple=None) -> None:
        if ylim:
            self.ylim = ylim


    def space(self, xlim: tuple=None, ylim: tuple=None, grid_size: int=None) -> None:
        if xlim and ylim:
            self.__limit(xlim, ylim)
        else:
            xlim = self.xlim
            ylim = self.ylim

        xlim = torch.tensor(xlim)
        ylim = torch.tensor(ylim)

        if not grid_size:
            grid_size = self._grid_size
    
        self.xlinspace = torch.linspace(xlim[0], xlim[1], grid_size)
        self.ylinspace = torch.linspace(ylim[0], ylim[1], grid_size)


    def build(self):
        raise NotImplementedError("Forward method must be implemented by the subclass.")