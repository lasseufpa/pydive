import torch
import numpy as np
import matplotlib.pyplot as plt

import copy

import os
import time

from pydive.utils import linalg as slinalg
from pydive.utils import training_routines

def optimization_path(self, start: tuple=None, model=None, trainloader=None, valloader=None,
                        optimizer=None, scheduler=None,
                        epochs=100, sample_freq=5, device='cpu', feed=False):
        # verfying parameters
        assert model is not None, "No model found for optimization"
        assert optimizer is not None, "No optimizer found for optimization"
        # adjust model in respect to start point
        if start:
            start = torch.tensor(start, dtype=torch.float64, device=device)
            w_adjusted = slinalg.adjust_weights(start[0], start[1])
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

        for t in range(epochs):
            # training 
            train_loss, train_acc = training_routines.train(model=model, loader=self.trainloader, optimizer=optimizer, device=device)
            # validation
            if t % sample_freq == sample_freq - 1 or t == epochs - 1:
                # validation
                val_loss, val_acc  = training_routines.validate(model=model, loader=self.valloader)
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

# def trajectory_path(model, optimizer, trainloader, valloader, criterion, base, u, v, nb_epochs=200, sample_freq=5, device='cpu', path='trajectory', load=True):
#     error_trajectory = []
#     loss_trajectory = []
#     np_error_trajectory = None
#     np_loss_trajectory = None

#     if load:
#         if path:
#             loss_trajectory = np.load(os.path.join(path, 'loss_trajectory.npy'))
#             error_trajectory = np.load(os.path.join(path, 'error_trajectory.npy'))
#         else:
#             if not os.path.exists(path):
#                 os.mkdir(path)
#             loss_trajectory = np.load(os.path.join(path, 'loss_trajectory.npy'))
#             error_trajectory = np.load(os.path.join(path, 'error_trajectory.npy'))
#     else:
#         tik = time.time()

#         loss, acc = training_routines.validate(model=model, loader=valloader, criterion=criterion, device=device)    
#         state = {'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}

#         if not os.path.exists(path):
#             os.mkdir(path)

#         torch.save(state, os.path.join(path, 'trajectory_0.pt'))
#         x0, y0 = slinalg.find_coordinates(state['model_state'], base, u, v)
#         loss_trajectory.append((x0, y0, loss))
#         error_trajectory.append((x0, y0, 100.0 - acc))
#         print(f"Época 0: Loss: {loss:>.5f}, Acc: {acc:>.2f}%, (x, y): {x0}, {y0}")
#         for t in range(nb_epochs):
#             train_loss, train_acc = training_routines.train(model=model, loader=trainloader, optimizer=model.optimizer, criterion=criterion, device=device)
#             if t % freq == 0 or t == nb_epochs - 1:
#                 loss, acc = training_routines.validate(model=model, loader=valloader, criterion=criterion, device=device)
                
#                 state = {'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}
#                 torch.save(state, os.path.join(path, 'trajectory_%d.pt' % (t+1)))

#                 x, y = slinalg.find_coordinates(state['model_state'], base, u, v)

#                 loss_trajectory.append((x, y, loss))
#                 error_trajectory.append((x, y, 100.0 - acc))
#                 if t < 10 or t == nb_epochs - 1:
#                     print(f"Época {t+1}: Loss: {loss:>.5f}, Acc: {acc:>.2f}%, (x, y): {x}, {y}")

#         np_error_trajectory = np.array(error_trajectory)
#         np_loss_trajectory = np.array(loss_trajectory)

#         error_path = os.path.join(path,'error_trajectory.npy')
#         loss_path = os.path.join(path,'loss_trajectory.npy')

#         np.save(loss_path, np_loss_trajectory)
#         np.save(error_path, np_error_trajectory)    

#         tok = time.time()
    
#     # print(f"Time for calculate trajectory: {(tok - tik)/60 if (tok - tik) > 60 else {tok - tik}}", end='')
#     # print("minutos" if (tok - tik) > 60 else "segundos")

#     return np_error_trajectory, np_loss_trajectory

# def plot_trajecctory_path(trajectory=None, xlim=(-1, 1), ylim=(-1, 1), show=False): 
#     assert trajectory is not None, "It's necessary to have a trajectory"
#     plt.scatter(trajectory[:, 0], trajectory[:, 1], zorder=2, c=trajectory[:,2])
#     plt.xlim(xlim[0], xlim[1])
#     plt.ylim(ylim[0], ylim[1])
#     if show:
#         plt.show()














