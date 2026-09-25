import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

class Attacker(ABC):
    def __init__(self, model, config, epsilon):
        """
        ## initialization ##
        :param model: Network to attack
        :param config : configuration to init the attack
        """
        self.config = config
        self.epsilon = epsilon
        self.model = model

    def __call__(self, x, y):
        x_adv = self.forward(x,y)
        return x_adv