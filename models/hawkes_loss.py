import numpy as np
import torch
from scipy.integrate import quad
import torch.nn as nn

class HawkesLoss(nn.Module):
    def __init__(self, trans_mtrx):
        super().__init__()
        self.trans_mtrx = trans_mtrx
        self.n1, self.n2 = 1, 1
        self.n_k = 4
        
        self.intensity_functions = None
        self.t_seq = None
        self.event_seq = None
        self.state_seq = None
        
        
    def get_time_intervals(self):
        return np.hstack([self.t_seq[:-1, np.newaxis], self.t_seq[1:, np.newaxis]])

    def forward(self, intensity_functions, t_seq, input_window):
        self.intensity_functions = intensity_functions
        self.t_seq = t_seq
        self.event_seq, self.state_seq = np.hsplit(input_window, 2)
        
        L1 = self.get_L1()
        L2 = self.get_L2()
        L3 = self.get_L3()
        hawkes_loss = L1 - self.n1*L2 + self.n2*L3 
        return hawkes_loss

    def get_L1(self):
        sum_L1 = 0
        t_intervals = self.get_time_intervals()
        for i, (t_begin, t_end) in enumerate(t_intervals):
            k = int(self.event_seq[i])
            intensity_new_k = self.intensity_functions[i + 1][k]
            old_intensity_functions = self.intensity_functions[i]
            total_intensity_integral = self.get_summed_intensity_integral(old_intensity_functions, t_begin, t_end)
            sum_L1 = sum_L1 + torch.log(intensity_new_k(t_end)) - total_intensity_integral
        return sum_L1

    def get_summed_intensity_integral(self, intensity_functions, t_begin, t_end):
        total_integral = 0
        for function in intensity_functions:
            total_integral = total_integral + quad(function, t_begin, t_end)[0]
        return total_integral

    def get_L2(self):
        sum_L2 = 0
        for i in range(len(self.event_seq) - 1):
            event = int(self.event_seq[i])
            tjplus1 = self.t_seq[i+1]
            event_intensity_function = self.intensity_functions[i][event]
            intesity_event = event_intensity_function(tjplus1)
            sum_L2 = sum_L2 - torch.log(intesity_event)
        return sum_L2

    def get_L3(self):
        sum_L3 = 0
        for i in range(0, len(self.event_seq) - 1):
            yiplus1 = int(self.event_seq[i + 1])
            xi = int(self.state_seq[i])
            xiplus1 = int(self.state_seq[i + 1])
            transition_prob = self.trans_mtrx[yiplus1][xi][xiplus1]
            sum_L3 = sum_L3 + np.log(transition_prob)
        return sum_L3
