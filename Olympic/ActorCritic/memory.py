import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class Memory:
    """Storing the memory of the trajectory (s, a, r ...)."""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.log_probs = []
        self.values = []

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.log_probs = []
        self.values = []