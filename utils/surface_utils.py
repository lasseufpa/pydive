import numpy as np
import os

def load_surface(path):
    return np.load(path)

def save_surface(path, file):
    np.save(os.path.join(path, file))