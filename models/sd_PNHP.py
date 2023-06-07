import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import quad
from scipy.integrate import quad
from torch.autograd import Variable
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from simple_linear_networks import *

class LSTMLayerStacked(nn.Module):
    def __init__(self, cell, input_size, hidden_size, t):
        super().__init__()
        self.stackedcell = LSTMStackedCells(cell, input_size, hidden_size)
        self.num_layers = 4
        self.hidden_size = hidden_size
        self.embed_layer = Embed_layer(input_size, hidden_size)
        self.h_input_merge_LC = Hidden_sharing_layer(5*hidden_size, 4*hidden_size)
        self.intensity_functions = []
        self.t_seq = t
        
    def forward(self, X, t_seq):
        
        first_run = True
        
        h1, h2, h3, h4 = [Variable(torch.randn(self.hidden_size)) for i in range(4)]
        c1_1, c2_1, c3_1, c4_1 = [Variable(torch.randn(self.hidden_size)) for i in range(4)]
        c1_2, c2_2, c3_2, c4_2 = [Variable(torch.randn(self.hidden_size)) for i in range(4)]
        
        i=0
        
        for x, tj in zip(X, self.t_seq):
            embedded_input = self.embed_layer(x)
            
            if not first_run: #TODO bit messy way to initialise run & set cell states. Clean it up if time left
#                 c1_1, c2_1, c3_1, c4_1 = self.fill_in_func(c1_func_list, tj)
#                 h1, h2, h3, h4 = self.fill_in_func(h_func_list, tj)
#                 c1_2, c2_2, c3_2, c4_2 = c2_list
                
                h1, h2, h3, h4 = [Variable(torch.randn(self.hidden_size)) for i in range(4)] #remove
                c1_1, c2_1, c3_1, c4_1 = [Variable(torch.randn(self.hidden_size)) for i in range(4)]
                c1_2, c2_2, c3_2, c4_2 = [Variable(torch.randn(self.hidden_size)) for i in range(4)]
                
#             h1, h2, h3, h4 = self.process_hidden_states_and_input(embedded_input, h1, h2, h3, h4)
            
            h1, h2, h3, h4 = [Variable(torch.randn(self.hidden_size)) for i in range(4)] #remove
            
            state = [(h1, c1_1, c1_2), (h2, c2_1, c2_2), (h3, c3_1, c3_2), (h4, c4_1, c4_2)]
            
            c2_list, c1_func_list, h_func_list, intensity_functions = self.stackedcell((state, tj))
            
            self.intensity_functions.append(intensity_functions)
            
            first_run = False
            
            print(f'LSTM LAYER RUN: {i}')
            i += 1
            
        return self.intensity_functions, t_seq
            
    def process_hidden_states_and_input(self, embedded_input, h1, h2, h3, h4):
        h_merged_with_input =  self.h_input_merge_LC(torch.concat([embedded_input, h1, h2, h3, h4]))
        
        h1, h2, h3, h4 = torch.split(h_merged_with_input, self.hidden_size)
    
        return h1, h2, h3, h4
    
            
    def fill_in_func(self, func, tj):
        f1, f2, f3, f4 = func[0](tj), func[1](tj), func[2](tj), func[3](tj)
        return (f1, f2, f3, f4)
    
class LSTMStackedCells(nn.Module):
    def __init__(self, cell, input_size, hidden_size):
        super().__init__()
        self.cell1 = cell(hidden_size, hidden_size)
        self.cell2 = cell(hidden_size, hidden_size)
        self.cell3 = cell(hidden_size, hidden_size)
        self.cell4 = cell(hidden_size, hidden_size)
        
    def forward(self, inputs):
        state, tj = inputs
        
        state1, state2, state3, state4 = state
        
        c1_2, c1_1_func, h1_func = self.cell1((state1, tj))
        c2_2, c2_1_func, h2_func = self.cell2((state2, tj))
        c3_2, c3_1_func, h3_func = self.cell3((state3, tj))
        c4_2, c4_1_func, h4_func = self.cell4((state4, tj))
        
        intensity_functions = self.get_intensity_functions()
        c2_list = [c1_2, c2_2, c3_2, c4_2]
        c1_func_list = [c1_1_func, c2_1_func, c3_1_func, c4_1_func]
        h_func_list = [h1_func, h2_func, h3_func, h4_func]
        
        return c2_list, c1_func_list, h_func_list, intensity_functions

    def get_intensity_functions(self): #using copies since the constants in object used for f() will be changing through time. TODO: add function and variable to a seperate dataclass that can be saved and isn't as clunky.
        f1 = self.cell1.get_frozen_intensity_function()
        f2 = self.cell2.get_frozen_intensity_function()
        f3 = self.cell3.get_frozen_intensity_function()
        f4 = self.cell4.get_frozen_intensity_function()
        return [f1, f2, f3, f4]



class LSTMCell(nn.Module):
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
        
        
    def forward(self, inputs):
        state, tj = inputs
        self.tj = tj
        
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
        
        return self.cy2, self.get_c, self.get_h
        
    def get_frozen_intensity_function(self): #variables are put in data_class so they become constants.
        intensity_dataclass = Intensity_func_and_constants(self.tj, self.cy1, self.cy2, self.decay_coef, self.o, self.LC_lambda)
        return intensity_dataclass.get_intensity_function()
        
    def get_intensity(self, t):
        h_t = self.get_h(t)
        intensity = self.LC_lambda(h_t)
        return intensity
        
    def get_h(self, t):
        c_t = self.get_c(t)
        h_t = self.o * self.Tanh(c_t)
        return h_t

    def get_c(self, t):
        c_t = self.cy2 + (self.cy1 - self.cy2)*torch.exp(-self.decay_coef*(t - self.tj))
        return c_t
    
class Intensity_func_and_constants():
    def __init__(self, tj, c1, c2, decay_coeff, o, LC_lambda):
        self.tj = tj
        self.cy1 = c1
        self.cy2 = c2
        self.decay_coef = decay_coeff       
        self.o = o
        self.LC_lambda = LC_lambda
        self.tanh = nn.Tanh()
            
    def get_intensity_function(self):
        return self.get_intensity
            
    def get_intensity(self, t):
        h_t = self.get_h(t)
        intensity = self.LC_lambda(h_t)
        return intensity
        
    def get_h(self, t):
        c_t = self.get_c(t)
        h_t = self.o * self.tanh(c_t)
        return h_t

    def get_c(self, t):
        c_t = self.cy2 + (self.cy1 - self.cy2)*torch.exp(-self.decay_coef*(t - self.tj))
        return c_t