import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pydive.bases import Izmailov
from pydive.utils.output_feed import format_time
from pydive.types import Number, Dict, Weights, Objective

import os
import csv
import math
import time
import copy
import random

import tabulate as tb
from typing import Callable

from collections import OrderedDict

from pydive.utils import surface_utils

from pydive.utils import linalg


class Surface(Izmailov):
    def __init__(self, 
                model: torch.nn.Module=None,
                loss_function: Objective=None, 
                loader: DataLoader=None,
                w1: Weights=None, w2: Weights=None, w3: Weights=None,
                device: str='cpu'
                ):
        # verifycation
        if not isinstance(device, str):
            raise ValueError("Device name must be a string")
        
        if model is None:
            raise ValueError("Model cannot be None.")
        elif not isinstance(model, torch.nn.Module):
            raise TypeError("Model is not nn.Module subclass.")
        
        if loss_function is None:
            raise ValueError("Loss function can't be None.")
        elif not isinstance(loss_function, (torch.nn.Module, Callable)):
            raise TypeError("Loss function is not an pydive.Objetive.")
        
        if DataLoader is None:
            raise ValueError("Dataloader cannot be None.")
        
        if w1 is None:
            raise ValueError("W1 can't be None")
        elif not isinstance(w1,  (Tensor, OrderedDict)):
            raise TypeError("W1 is not an Weight-like")
        if w2 is None:
            raise ValueError("W2 can't be None")
        elif not isinstance(w2,  (Tensor, OrderedDict)):
            raise TypeError("W2 is not an Weight-like")
        if w3 is None:
            raise ValueError("W3 can't be None")
        elif not isinstance(w3,  (Tensor, OrderedDict)):
            raise TypeError("W3 is not an Weight-like")
        
        # Izmailov's base init
        super().__init__(w1=w1, w2=w2, w3=w3, device=device)
        # setting surface device
        self.device = device
        # set model
        self._set_model(model.to(device))
        # set dataloader
        self._set_loader(loader)
        # set loss funtion
        self._set_loss_function(loss_function)


    def _set_model(self, new_model: torch.nn.Module=None) -> tuple:
        if isinstance(new_model, torch.nn.Module):        
            self.backup_model = new_model
            self.model = copy.deepcopy(self.backup_model)
        else:
            raise ValueError(f"Trying to insert an invalid model object: {type(model)}")


    def _set_loss_function(self, loss_function: Objective=None) -> None:
        if isinstance(loss_function, (torch.nn.Module, Callable)):
            self.loss_function = loss_function
        else:   
            raise ValueError(f"Trying to insert an invalid loss function object: {type(loss_function)}")
        
    
    def _set_loader(self, loader: DataLoader=None) -> None:
        if isinstance(loader, DataLoader):
            self.loader = loader
        else:   
            raise ValueError(f"Trying to insert an invalid dataloader: {type(loader)}")


    def __call__(self, *args, **kwargs):
        self.build(*args, **kwargs)
    

    def build(self, loader: DataLoader=None, surface: str='loss', feed: bool=False) -> None:
        if not loader:
            loader = self.valloader
        self.surface = []
        
        tik = time.time() 
        for x in self.xlinspace:
            for y in self.ylinspace:
                if surface == 'loss':
                    point = self._loss_from_point(x=x, y=y, loader=loader, feed=feed)
                    self.surface.append(point[:3])  
                elif surface == 'error':
                    point = self._error_from_point(x=x, y=y, loader=loader, feed=feed)
                    self.surface.append(point[:3])
                       
        tok = time.time()
        self.build_time = tok - tik
        self.surface = np.array(self.surface)
        if feed:
            print(f'Time for creating surface: {(self.build_time):.2f} segundos')
        

    def _loss_from_point(self, x: Number, y: Number, loader: DataLoader, feed: bool=False) -> tuple:
        tik = time.time()
        adjusted_weights = self.adjust_weights(x, y)
        model = self.model
        model.load_state_dict(adjusted_weights)
        loss, _  = self._validate(model=model, loader=loader)
        tok = time.time()
        if feed:
            print(f"Point calculated: [{x:>.5f},{y:>.5f}], Loss: [{loss:>.4f}], Time: {format_time(tok - tik)}")
        return (x, y, loss)
    

    def _error_from_point(self, x: Number, y: Number, loader: DataLoader, feed: bool=False) -> tuple:
        tik = time.time()
        adjusted_weights = self.adjust_weights(x, y)
        model = self.model
        model.load_state_dict(adjusted_weights)
        _, acc  = self._validate(model=model, loader=loader)
        error = 100.0 - acc
        tok = time.time()
        if feed:
            print(f"Point calculated: [{x:>.5f},{y:>.5f}], Loss: [{error:>.4f}%], Time: {format_time(tok - tik)}")
        return (x.item(), y.item(), error)


    def _validate(self, model: nn.Module=None, loader: DataLoader=None, loss_function: Objective=None, device: str=None) -> tuple:
        if not model:
            model = self.model
        if not loader:
            loader = self.valloader
        if not device:
            device = self.device
        if not loss_function:
            loss_function = self.loss_function

        loader_size = len(loader.dataset)
        num_batches = len(loader)

        model.eval()

        total_loss = 0.
        correct = 0.

        with torch.no_grad():
            for  X, y in loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                total_loss +=  loss_function(output, y).item()
                correct += (output.argmax(1) == y).type(torch.float64).sum().item()

        avg_loss = total_loss / num_batches
        accuracy = (correct / loader_size)*100

        return avg_loss, accuracy
    

    def save_surface(self, path) -> None:
        np_surface = np.array(self.surface)
        np.save(path, np_surface)    


    def load_surface(self, path) -> None:
       self.surface = np.load(path)


    def grid(self):
        self.x_grid, self.y_grid = np.meshgrid(np.unique(self.surface[:, 0]), np.unique(self.surface[:, 1]))
        self.loss_grid = self.surface[:, 2].reshape(self.x_grid.shape)


    def level_mapping(self, x):
        return np.power(x, 2)
    

    def fit_levels(self, num_levels=8, init=None):
        self.min_loss = np.min(self.surface[:,2])
        self.max_loss = np.max(self.surface[:,2])
        self.level_limits = [self.init] if init else [self.min_loss]

        for i in range(1, num_levels+1):
            x = i/(num_levels-1)
            self.level_limits.append((self.level_limits[-1] + self.level_mapping(x)))

        return np.array(self.level_limits)
    
    def plot(self, surface=None, projection='2d', xlim=None, ylim=None, zlim=None, num_levels=8, save=None, cmap='Spectral'):
        if not surface:
            surface = self.surface
        else:
            if not (type(surface).__module__ == np.__name__):
                surface = np.array(surface)
        if not xlim:
            xlim = self.xlim
        if not ylim:
            ylim = self.ylim

        self.x_grid, self.y_grid = np.meshgrid(np.unique(surface[:, 0]), np.unique(surface[:, 1]))
        self.loss_grid = surface[:, 2].reshape(self.x_grid.shape).T
        levels = self.fit_levels(num_levels=num_levels)

        if not zlim:
            zlim = (levels[0], levels[-1])

        if projection == '3d':
            self.loss_3dgrid = np.copy(self.loss_grid)
            outside_limits = np.logical_or(self.loss_grid < zlim[0], self.loss_grid > zlim[1])
            self.loss_3dgrid[outside_limits] = np.nan
            ax = plt.axes(projection='3d')
            contourf = ax.plot_surface(self.x_grid, self.y_grid, self.loss_3dgrid, vmin=self.loss_grid.min(), vmax=10, cmap=cmap, zorder=1)
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])
            ax.set_zlim(zlim[0], zlim[1])
        else:
            colors = ['#b97373', '#ffae73', '#fff173', '#c7ffa6', '#8fffde', '#8fffde', '#8fffde']
            contourf = plt.contourf(self.x_grid, self.y_grid, self.loss_grid, levels=levels, cmap=cmap, extend='both', zorder=0)
            plt.contour(self.x_grid, self.y_grid, self.loss_grid,  levels=levels, extend='both', cmap=cmap, zorder=1)
            plt.xlim(xlim[0], xlim[1])
            plt.ylim(ylim[0], ylim[1])
        plt.colorbar(contourf)
        

        plt.xlabel('x')
        plt.ylabel('y')

        self.num_levels = num_levels

        if save:
            plt.savefig(f"{str(time.time())}.png")
            

class OnlineSurface(Surface):
    def __init__(self, 
                model: torch.nn.Module=None,
                loss_function: Objective=None, 
                loader: DataLoader=None,
                w1: Weights=None, w2: Weights=None, w3: Weights=None,
                device: str='cpu'
                ):
        super().__init__(model=model, loss_function=loss_function,
                         loader=loader,
                         w1=w1, w2=w2, w3=w3,
                         device=device)


    def _set_loader(self, loader=None) -> None:
        if loader:
            self.loader = loader
        else:   
            raise ValueError(f"Trying to insert an invalid dataloader: {type(loader)}")
        

    def save_surface(self, path) -> None:
        np_surface = np.array(self.surface)
        surface_utils.save_surface(path, np_surface)  


    def load_surface(self, path) -> None:
       self.surface = surface_utils.load_surface(path)


    def build(self, loader: DataLoader=None, surface: str='preq_error', device: str=None, feed: bool=False) -> None:
        if not loader:
            loader = self.valloader
        if not device:
            device = self.device

        self.surface = []
        
        tik = time.time() 
        for x in self.xlinspace:
            for y in self.ylinspace:
                if surface == 'preq_error':
                    point = self._preq_error_from_point(x=x, y=y, loader=loader, feed=feed)
                elif surface == 'preq_acc':
                    point = self._preq_acc_from_point(x=x, y=y, loader=loader, feed=feed)
                self.surface.append(point[:3])  

        tok = time.time()
        self.build_time = tok - tik
        self.surface = np.array(self.surface)
        if feed:
            print(f'Time for creating surface: {format_time(self.build_time)} segundos')


    def _preq_error_from_point(self, x: Number, y: Number, loader: DataLoader, device: str=None, feed: bool=False):
        if not device:
            device = self.device
        tik = time.time()
        adjusted_weights = linalg.adjust_by_coordinates(self.w1, self.xdirection, self.ydirection, x, y)
        model = self.model
        self.load_state_s3wa(model, adjusted_weights)
        preq_error  = self._validate(model=model, loader=loader, func='preq_error', device=device)
        tok = time.time()
        if feed:
            print(f"Point calculated: [{x:>.5f},{y:>.5f}], Error: [{preq_error:>.4f}], Time: {format_time(tok - tik)}")
    
        return (x, y, 100*preq_error)   
    

    def _preq_acc_from_point(self, x: Number, y: Number, loader: DataLoader, device: str=None, feed: bool=False):
        if not device:
            device = self.device
        tik = time.time()
        adjusted_weights = linalg.adjust_by_coordinates(self.w1, self.xdirection, self.ydirection, x, y)
        model = self.model
        self.load_state_s3wa(model, adjusted_weights)
        preq_acc  = self._validate(model=model, loader=loader, func='preq_acc', device=device)
        tok = time.time()
        if feed:
            print(f"Point calculated: [{x:>.5f},{y:>.5f}], Accuracy: [{preq_acc:>.4f}], Time: {format_time(tok - tik)}")
    
        return (x, y, 100*preq_acc)   
    
    
    def _loss_from_point(self, x: Number, y: Number, loader: DataLoader, feed: bool=False) -> tuple:
        tik = time.time()
        adjusted_weights = self.adjust_weights(x, y)
        model = self.model
        self.load_state_s3wa(model, adjusted_weights)
        loss, _  = self._validate(model=model, loader=loader)
        tok = time.time()
        if feed:
            print(f"Point calculated: [{x:>.5f},{y:>.5f}], Loss: [{loss:>.4f}], Time: {format_time(tok - tik)}")
    
        return (x.item(), y.item(), loss)
    
    
    def _error_from_point(self, x: Number, y: Number, loader: DataLoader, feed: bool=False) -> tuple:
        tik = time.time()
        adjusted_weights = self.adjust_weights(x, y)
        model = self.model
        self.load_state_s3wa(model, adjusted_weights)
        _, acc  = self._validate(model=model, loader=loader)
        error = 100.0 - acc
        tok = time.time()
        if feed:
            print(f"Point calculated: [{x:>.5f},{y:>.5f}], Loss: [{error:>.4f}%], Time: {format_time(tok - tik)}")
    
        return (x.item(), y.item(), error)


    def load_state_s3wa(self, model, state_dict: OrderedDict=None) -> None:
        base_autoencoder_state = model.autoencoder.state_dict()
        base_enc, base_dec = self.split_state_dict(base_autoencoder_state, 'encoder.')
        new_enc, new_mlp = self.split_state_dict(state_dict, 'encoder.')
        new_autoencoder = OrderedDict(**new_enc, **base_dec)
        model.autoencoder.load_state_dict(new_autoencoder)
        model.model.load_state_dict(new_mlp)

    
    def split_state_dict(self, state_dict: OrderedDict, key: str=None) -> OrderedDict:
        w_key, w_left = OrderedDict(), OrderedDict()
        if key == None:
            return copy.deepcopy(state_dict), copy.deepcopy(state_dict)
        else:
            for k in state_dict.keys():
                if key in k:
                    w_key[k] = copy.deepcopy(state_dict[k])
                else:
                    w_left[k] = copy.deepcopy(state_dict[k])

        return w_key, w_left
    

    def prequential_error_with_fading(self, predictions, true_labels, fading_factor=0.999):
        """Prequential error with fading factor for a batch of predictions"""
        preq_incorrect = 0
        preq_total = 0

        for pred, true in zip(predictions, true_labels):
            preq_incorrect = int(pred != true) + fading_factor * preq_incorrect
            preq_total = 1 + fading_factor * preq_total
            running_error = preq_incorrect / preq_total if preq_total != 0 else 0
        
        return running_error
    
    
    def prequential_accuracy_with_fading(self, predictions, true_labels, fading_factor=0.999):
        """Prequential error with fading factor for a batch of predictions"""
        preq_correct = 0
        preq_total = 0

        for pred, true in zip(predictions, true_labels):
            preq_correct = int(pred == true) + fading_factor * preq_correct
            preq_total = 1 + fading_factor * preq_total
            running_acc = preq_correct / preq_total if preq_total != 0 else 0
        
        return running_acc
    
    
    def _validate(self, model: nn.Module=None, loader: DataLoader=None, func=None, device: str=None) -> tuple:
        if not model: model = self.model
        if not loader: loader = self.valloader
        if not device: device = self.device
        if not func: func = 'preq_error'

        model.eval()
        
        if hasattr(model, 'out_act'):
            out_act = model.out_act
        else:
            out_act = nn.Softmax(dim=1)

        preds = torch.empty(0).to(device)
        true_labels = torch.empty(0).to(device)
            
        with torch.no_grad():
            for time_step, (inputs, targets) in enumerate(loader):
                # clear the gradients, we clear them for each instance - strict online
                        
                last_input = inputs[-1].unsqueeze(0)
                last_target = targets[-1].unsqueeze(0)
                # remove mask column
                last_input = last_input[:, :-1].float()

                x_enc = model.autoencoder.encoder(last_input.float())
                x_enc = x_enc.detach() 

                yhat = model(x_enc) 
                    
                true_labels = torch.cat((true_labels, last_target))

                preds = torch.cat((preds, torch.argmax(out_act(yhat.detach()), axis=1)))

            if func == 'preq_error':
                return self.prequential_error_with_fading(preds, true_labels)
            elif func == 'preq_acc':
                return self.prequential_accuracy_with_fading(preds, true_labels)
    
    
    def save_surface(self, path) -> None:
        np_surface = np.array(self.surface)
        np.save(path, np_surface)    


    def load_surface(self, path):
       self.surface = np.load(path)


    def grid(self):
        self.x_grid, self.y_grid = np.meshgrid(np.unique(self.surface[:, 0]), np.unique(self.surface[:, 1]))
        self.loss_grid = self.surface[:, 2].reshape(self.x_grid.shape).T


    def optimization_path(self, start=None, model=None, optimizer=None,
                   limit=100, freq=5, device=None, feed=False):
        # verifying 
        if not optimizer or not isinstance(optimizer, torch.optim):
            assert self.__optimization is not None and isinstance(self.optimizer, torch.optim), "No optimizer passed to function or optimizer don't match Pytoch nn.optim module, same for surface optimizer"
            optimizer = self.optimizer
        # setting devcie
        if not device: 
            device = self.device
        # verfying parameters
        if not model or not isinstance(model, nn.Module):
            print("Warning: Model not found, using surface defaults...")
            model = copy.deepcopy(self.model)
        # adjust model in respect to start point
        if start:
            start = torch.tensor(start, dtype=torch.float64, device=device)
            w_adjusted = self.adjust_weights(start[0], start[1])
            model.load_state_dict(w_adjusted)
            model.to(device)
            # updating optimizer parameters
        else:
            start = self.find_coordinates(model.state_dict())
        # adjust optimizer
        for group in optimizer.param_groups:
            group['params'] = list(model.parameters())

        w_start = self.find_coordinates(model.state_dict())
        # results lists
        self.error_optimization = list()
        self.loss_optimization = list()
        self.epoch_path = list()
        self.epoch_path.append(0)
        # inital state
        val_loss, val_acc = self._validate(model, self.valloader)
        # append initial state
        self.loss_optimization.append((start[0], start[1], val_loss))
        self.error_optimization.append((start[0], start[1], 100 - val_acc))
        # feeding initial state
        if feed:
            print(f"Starting point: {start}")
            print(f"Initial state: Loss: {val_loss:>.5f}, Acc: {val_acc:>.2f}%, (x, y): {w_start}")
            print(w_start[0].item(), w_start[1].item())

        for t in range(limit):
            # training 
            train_loss, train_acc = self._train(model=model, loader=self.trainloader, optimizer=optimizer, device=device)
            # validation
            if t % freq == freq - 1 or t == limit - 1:
                # validation
                val_loss, val_acc  = self._validate(model=model, loader=self.valloader)
                # model and optimizer state
                # state = {'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}
                # find current coordinates
                x, y = self.find_coordinates(model.state_dict())
                # append results to optimization
                self.loss_optimization.append((x, y, val_loss))
                self.error_optimization.append((x, y, 100.0 - val_acc))
                self.epoch_path.append(t)
                # feeding current results
                if feed:
                    print(f"Epoch {t+1}: Loss: {val_loss:>.15f}, Acc: {val_acc:>.2f}%, (x, y): ({x:>.17f}, {y:>.17f})")
        if self.device != 'cpu':
            self.loss_optimization = np.array([(x.cpu(), y.cpu(), l.cpu()) for x, y, l in self.loss_optimization])
            self.error_optimization = np.array([(x.cpu(), y.cpu(), l.cpu()) for x, y, l in self.error_optimization])
        else:
            self.loss_optimization = np.array(self.loss_optimization)
            self.error_optimization = np.array(self.error_optimization)
        self.epoch_path = np.array(self.epoch_path)

        return copy.deepcopy(self.loss_optimization), copy.deepcopy(self.error_optimization), copy.deepcopy(self.epoch_path)
    
    def level_mapping(self, x):
        return np.power(x, 2)
    
    def notable_points(self, loader=None):
        if not loader:
            loader = self.valloader

        notable_points = list()
        test_model = copy.deepcopy(self.backup_model)
        # W1 point
        test_model.model.load_state_dict(self.w1)
        x, y = self.find_coordinates(self.w1)
        notable_points.append((x, y,self._validate_preq_error(test_model, loader)))
        # W2 point
        test_model.model.load_state_dict(self.w2)
        x, y = self.find_coordinates(self.w2)
        notable_points.append((x, y,self._validate_preq_error(test_model, loader)))
        # W3 point
        test_model.model.load_state_dict(self.w3)
        x, y = self.find_coordinates(self.w3)
        notable_points.append((x, y, self._validate_preq_error(test_model, loader)))

        return np.array(notable_points) 

    def fit_levels(self, bottom: float=None, top: float=None, num_levels: int=8, fit_type: str='linear', factor=2):
        if not bottom:
            bottom = np.min(self.surface[:,2])
        if not top:
            top = np.max(self.surface[:,2])

        level_limits = [bottom]

        if fit_type == 'linear':
            factor = (top - bottom) / (num_levels-1)
            for i in range(1, num_levels):
                level_limits.append(level_limits[i-1] + factor)

            return np.array(level_limits)
        
        elif fit_type == 'exp':
            level_limits = self.custom_exponential_scale(bottom, top, num_levels, factor=factor)

            return level_limits

    def custom_exponential_scale(self, a, b, num_levels=8, factor=2.0, initial_factor=3.0, smooth='min'):
        # Apply an initial transformation to create a smoother start
        smooth_start_space = np.logspace(0, 1, num_levels, base=initial_factor)
        
        # Normalize the smooth_start_space to be between 0 and 1
        smooth_start_space = (smooth_start_space - smooth_start_space.min()) / (smooth_start_space.max() - smooth_start_space.min())

        if smooth == 'min':
            exp_space = a + (b - a) * (smooth_start_space ** factor)
        elif smooth == 'max':
            # Invert the normalization for smooth scaling near the maximum       
            smooth_start_space = 1 - smooth_start_space
            # Apply the main exponential transformation
            exp_space = a + (b - a) * (smooth_start_space ** factor)
            exp_space = np.flip(np.append(exp_space, 0.))

        else:
            exp_space = a + (b - a) * (smooth_start_space ** factor)
            exp_space = np.append(exp_space, 100.)

        return exp_space
    
    def focused_interval_scaling(self, focus: int=None, interval: int=None, top: float=None, bottom: float=None, num_levels: int=8):
        """Create a level scaling which a interval in focused"""
        if not top: top = np.max(self.surface[:,2])
        if not bottom: bottom = np.min(self.surface[:,2])

        if not focus: focus = num_levels // 2

        try:
            interval = np.sort(np.array(interval))
        except TypeError:
            return self.fit_levels(bottom, top, num_levels)
        
        sub_interval_1 = [bottom]
        sub_interval_2 = [interval[0]]
        sub_interval_3 = [interval[-1]]

        s1 = (num_levels - focus) // 2
        alpha_1 = abs(interval[0] - bottom) / s1
        s2 = focus
        alpha_2 = abs(interval[-1] - interval[0]) / s2
        s3 = num_levels - (s2 + s1)
        alpha_3 = abs(top - interval[-1]) / s3
        
        for i in range(1, s1+1):
            sub_interval_1.append(sub_interval_1[-1] + alpha_1)
        
        for i in range (1, s2+1):
            sub_interval_2.append(sub_interval_2[-1] + alpha_2)

        for i in range (1, s3+1):
            sub_interval_3.append(sub_interval_3[-1] + alpha_3)

        level_limits = sub_interval_1[0:-1] + sub_interval_2[0:-1] + sub_interval_3

        return level_limits

    def plot(self, surface=None, projection='2d', axis=None, metric='preq_error',
            bottom=None, top=None, num_levels=8, 
            focused=False, focus=None, interval=None, level_fill='linear', factor=2, initial_factor=1.5, smooth = 'min',
            xlim=None, ylim=None, zlim=None,
            save=None,
            cmap='Spectral',
            fill=True
            ):
        
        if not surface: 
            surface = self.surface
        else:
            if not (type(surface).__module__ == np.__name__):
                surface = np.array(surface)

        if not xlim: xlim = self.xlim
        if not ylim: ylim = self.ylim

        if smooth == 'min':
            if not bottom: 
                bottom = np.min(self.surface[:,2])
            if not top: 
                top = int(np.max(self.surface[:,2]))

        elif smooth == 'max':
            if not bottom: 
                bottom = int(np.min(self.surface[:,2]))
            if not top: 
                top = np.max(self.surface[:,2])

        self.x_grid, self.y_grid = np.meshgrid(np.unique(surface[:, 0]), np.unique(surface[:, 1]))
        self.loss_grid = surface[:, 2].reshape(self.x_grid.shape).T

        if focused:
            if not focus: focus = num_levels // 2
            levels = self.focused_interval_scaling(focus=focus, interval=interval, top=top, bottom=bottom, num_levels=num_levels)
        
        else:
            levels = self.custom_exponential_scale(bottom, top, num_levels, factor=factor, initial_factor=initial_factor, smooth='max')
        print(levels)
        if not zlim:
            zlim = (levels[0], levels[-1])

        if projection == '2d':
            colors = ['#c26c6b', '#f99758', '#ffcb37', '#f0f43e', '#baff8a', '#82fec8', '#61e9f5', '#6f83fe'] # original pavel's colorset 
            colors = ['#bf6e6e', '#ffa95a', '#fff040', '#c1ff92', '#79ffdd', '#62ceff', '#7183ff', '#7373c2'] # saturated colorset
            boundaries_colors = ['#ff2a29', '#ff9600', '#cbfc05', '#6bffbc', '#00e4fd', '#00e4fd', '#006bff', '#4848f0']
            if smooth == 'max':
                colors.reverse()
                boundaries_colors.reverse()

            boundaries_cmap = mcolors.ListedColormap(boundaries_colors)
            boundaries_norm = mcolors.BoundaryNorm(levels[1:], ncolors=len(levels))

            if fill: 
                cmap = mcolors.LinearSegmentedColormap.from_list('fill_cmap', colors, N=len(levels)-1)
                norm = mcolors.BoundaryNorm(levels, cmap.N)
                contourf = axis.contourf(self.x_grid, self.y_grid, self.loss_grid, levels=levels, cmap=cmap, norm=norm, zorder=0)

            axis.contour(self.x_grid, self.y_grid, self.loss_grid,  levels=levels, cmap=boundaries_cmap, norm=boundaries_norm, zorder=1)
            axis.set_xlim(xlim[0], xlim[1])
            axis.set_ylim(ylim[0], ylim[1])

        if fill:
            fig = plt.gcf()
            if smooth == 'min':
                levels_list = levels.tolist()
                str_levels = [f'{level:.2f}' for level in levels_list[:-1]] + ['> ' + str(int(levels_list[-2]))]
                print(str_levels)
                colobar = fig.colorbar(contourf, spacing='uniform', extend='max', extendrect=True, extendfrac='auto', ticks=levels_list)
                colobar.set_ticklabels(str_levels)
                colobar.outline.set_visible(False)
                colobar.ax.tick_params(axis='both', which='both', size=0, pad=7)

            elif smooth == 'max':
                levels_list = levels.tolist()
                str_levels = ['< ' + str(levels_list[1])] + [f'{level:.2f}' for level in levels_list[1:]]
                print(str_levels)
                colobar = fig.colorbar(contourf, spacing='uniform', extend='max', extendrect=True, extendfrac='auto', ticks=levels_list)
                colobar.set_ticklabels(str_levels)
                colobar.outline.set_visible(False)
                colobar.ax.tick_params(axis='both', which='both', size=0, pad=7)    

        self.num_levels = num_levels

        if save:
            plt.savefig(f"{str(time.time())}.png")
