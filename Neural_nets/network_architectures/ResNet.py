import torch
from torch import nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, amount_of_input_neurons, amount_of_output_neurons, amount_of_hidden_layers):
        super().__init__()
        amount_of_hidden_neurons = 4*amount_of_input_neurons
        self.amount_of_hidden_layers = amount_of_hidden_layers
        self.lin_input = nn.Linear(amount_of_input_neurons, amount_of_hidden_neurons)
        self.lin = [nn.Linear((i+1)*amount_of_hidden_neurons, amount_of_hidden_neurons) for i in range(amount_of_hidden_layers)]
        self.lin_output = nn.Linear((amount_of_hidden_layers+1)*amount_of_hidden_neurons, amount_of_output_neurons)

    def forward(self, xb):
        xb_hidden = [F.leaky_relu(self.lin_input(xb))]
        for i in range(self.amount_of_hidden_layers):
            xb_hidden.append(F.leaky_relu(self.lin[i](torch.cat(xb_hidden,-1))))
        return F.softmax(self.lin_output(torch.cat(xb_hidden,-1)))