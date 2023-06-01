import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.jit as jit
import warnings
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor
import numbers
import numpy as np

class LSTMCellUnknownData(nn.Module):
    def __init__(self, input_size, hidden_size = 16):
        super().__init__()
        self.Tanh = nn.Tanh()
        self.Softplus = nn.Softplus()
        
        self.LC_i1 = Gate_network(input_size, hidden_size)
        self.LC_i2 = Gate_network(input_size, hidden_size)
        self.LC_f1 = Gate_network(input_size, hidden_size)
        self.LC_f2 = Gate_network(input_size, hidden_size)
        self.LC_o = Gate_network(input_size, hidden_size)
        self.LC_z = Gate_network(input_size, hidden_size)
        self.LC_d = Decay_network(input_size, hidden_size)
        self.LC_lambda = Decode_layer(input_size, hidden_size)
        
        self.tj = None
        self.cy1 = None
        self.cy2 = None
        self.decay_coef = None
        self.o = None
        
    def get_intensity(self, t):
        h_t = self.get_h(t)
        intensity = self.LC_lambda(h_t)
        
    def get_h(self, t):
        c_t = self.calc_c(t)
        h_t = self.o * torch.tanh(c_t)
        return h_t

    def get_c(self, t):
        c_t = self.cy2 + (self.cy1 - self.cy2)*torch.exp(-self.decay_coef*(t - tj))
        return c_t
        
    def forward(self, state):
        self.check_if_tj_is_set()
        
        hx, cx1, cx2 = state
        
        i1 = self.LC_i1(hx)
        i2 = self.LC_i2(hx)
        f1 = self.LC_f1(hx)
        f2 = self.LC_f2(hx)
        o = self.LC_o(hx)
        z = self.Tanh(self.LC_z(hx))

        self.o = o
        self.cy1 = (f1 * cx1) + (i1 * z)
        self.cy2 = (f2 * cx2) + (i2 * z)
        self.decay_coef = self.LC_d(hx)
        
        return self.cy2
    
    def set_tj(self, tj):
        self.tj = tj
        
    def check_if_tj_is_set(self):
        if self.tj == None:
            raise Exception('Cannot run forward of LSTM cell before its starting time is set. Use set_tj to set tj.')
            
class LSTMCellKnownData(nn.Module):
    def __init__(self, time_interval, input_size, hidden_size = 16):
        super().__init__()
        self.Tanh = nn.Tanh()
        self.Softplus = nn.Softplus()
        
        self.LC_i1 = Gate_network(input_size, hidden_size)
        self.LC_i2 = Gate_network(input_size, hidden_size)
        self.LC_f1 = Gate_network(input_size, hidden_size)
        self.LC_f2 = Gate_network(input_size, hidden_size)
        self.LC_o = Gate_network(input_size, hidden_size)
        self.LC_z = Gate_network(input_size, hidden_size)
        self.LC_d = Decay_network(input_size, hidden_size)
        self.LC_lambda = Decode_layer(input_size, hidden_size)
        
        self.t_start, self.t_end = time(interval)
        self.cy1 = None
        self.cy2 = None
        self.decay_coef = None
        self.o = None
        
    def get_intensity(self, h_t):
        intensity = self.LC_lambda(h_t)
        return intensity
        
    def get_h(self, c_t):
        h_t = self.o * torch.tanh(c_t)
        return h_t

    def get_c(self):
        c_t = self.cy2 + (self.cy1 - self.cy2)*torch.exp(-self.decay_coef*(self.t_end - self.t_start))
        return c_t
        
    def forward(self, state):
        self.check_if_tj_is_set()
        
        hx, cx1, cx2 = state
        
        i1 = self.LC_i1(hx)
        i2 = self.LC_i2(hx)
        f1 = self.LC_f1(hx)
        f2 = self.LC_f2(hx)
        o = self.LC_o(hx)
        z = self.Tanh(self.LC_z(hx))

        self.o = o
        self.cy1 = (f1 * cx1) + (i1 * z)
        self.cy2 = (f2 * cx2) + (i2 * z)
        self.decay_coef = self.LC_d(hx)
        
        c_output = self.get_c()
        h_output = self.get_h(c_output)
        intensity_output = self.get_intensity(h_output)
        
        state = (h_output, c_output, cy2)
        
        return intensity_output, state
    
    def set_tj(self, tj):
        self.tj = tj
        
    def check_if_tj_is_set(self):
        if self.tj == None:
            raise Exception('Cannot run forward of LSTM cell before its starting time is set. Use set_tj to set tj.')

class Gate_network(nn.Module):
    def __init__(self, input_size, hidden_size = 16):
        super().__init__()
        self.Tanh = nn.Tanh()
        self.Sigmoid = nn.Sigmoid()
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, input: Tensor):
        x = self.Tanh(self.layer1(input))
        x = self.Tanh(self.layer2(x))
        return self.Sigmoid(x)
    
class Decay_network(nn.Module):
    def __init__(self, input_size, hidden_size = 16):
        super().__init__()
        self.Softplus = nn.Softplus()
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, input: Tensor):
        x = self.Softplus(self.layer1(input))
        x = self.Softplus(self.layer2(x))
        return x
    
class Decode_layer(nn.Module):
    def __init__(self, input_size, hidden_size = 16):
        super().__init__()
        self.Softplus = nn.Softplus()
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        
    def forward(self, input: Tensor):
        x = self.Softplus(self.layer1(input))
        x = self.Softplus(self.output_layer(x))
        return x

class Embed_layer(nn.Module):
    def __init__(self, input_size, hidden_size = 16):
        super().__init__()
        self.Tanh = nn.Tanh
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(input_size, hidden_size)
        
    def forward(self, input: Tensor):
        x = self.Tanh(self.layer1(input))
        x = self.Tanh(self.layer2(x))
        return x