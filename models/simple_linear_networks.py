import torch.nn as nnsubmodules
from torch import Tensor
import torch.nn as nn

class Gate_network(nn.Module):
    def __init__(self, input_size, hidden_size = 16):
        super().__init__()
        self.tanh = nn.Tanh()
        self.Sigmoid = nn.Sigmoid()
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, input: Tensor):
        x = self.tanh(self.layer1(input))
        x = self.tanh(self.layer2(x))
        return self.Sigmoid(x)
    
class Decay_network(nn.Module):
    def __init__(self, input_size, hidden_size = 16):
        super().__init__()
        self.softplus = nn.Softplus()
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, input: Tensor):
        x = self.softplus(self.layer1(input))
        x = self.softplus(self.layer2(x))
        return x
    
class Decode_layer(nn.Module):
    def __init__(self, input_size, hidden_size = 16):
        super().__init__()
        self.softplus = nn.Softplus()
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        
    def forward(self, input: Tensor):
        x = self.softplus(self.layer1(input))
        x = self.softplus(self.output_layer(x))
        return x

class Embed_layer(nn.Module):
    def __init__(self, input_size, hidden_size = 16):
        super().__init__()
        self.tanh = nn.Tanh()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, input: Tensor):
        x = self.tanh(self.layer1(input))
        x = self.tanh(self.layer2(x))
        return x
    
class Hidden_sharing_layer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.tanh = nn.Tanh()
        self.layer1 = nn.Linear(input_size, output_size)
        self.layer2 = nn.Linear(output_size, output_size)
        
    def forward(self, input: Tensor):
        x = self.tanh(self.layer1(input))
        x = self.tanh(self.layer2(x))
        return x