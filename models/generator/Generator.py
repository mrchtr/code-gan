from random import randint

from torch import nn
import torch.nn.functional as f
import torch
from models.generator.RelationalMemory import RelationalMemory


class Generator(nn.Module):
    """
    Boilerplate class for different generator models.
    """
    def __init__(self, config):
        super(Generator, self).__init__()
        self.temperature = config.temperature

        self.device = config.device


    @staticmethod
    def add_gumbel(o_t, device, eps=1e-10):
        """Add o_t by a vector sampled from Gumbel(0,1)"""
        u = torch.zeros(o_t.size())
        u = u.to(device)

        u.uniform_(0, 1)
        g_t = -torch.log(-torch.log(u + eps) + eps)
        gumbel_t = o_t + g_t
        return gumbel_t