import torch.nn as nn
import torch
import torch.nn.functional as F

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input, hidden):
        hidden = F.tanh(self.i2h(input) + self.h2h(hidden)) # tanh activation to make sure stuff is between -1 and 1
        output = self.h2o(hidden) # new hidden used to calculate output
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)