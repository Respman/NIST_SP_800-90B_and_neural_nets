from torch import nn
from collections import OrderedDict


def FNN(ctx):
    va_params = []
    if ctx["model"]["amount_hidden_neurons"] != 0:
        va_params.append(nn.Linear(ctx["model"]["amount_of_input_neurons"], ctx["model"]["amount_hidden_neurons"][0]))
        va_params.append(nn.LeakyReLU(0.01))
        for i in range(len(ctx["model"]["amount_hidden_neurons"])-1):
            va_params.append(nn.Linear(ctx["model"]["amount_hidden_neurons"][i], ctx["model"]["amount_hidden_neurons"][i+1]))
            va_params.append(nn.LeakyReLU(0.01))
        va_params.append(nn.Linear(ctx["model"]["amount_hidden_neurons"][-1], ctx["model"]["amount_of_output_neurons"]))
    else:
        va_params.append(nn.Linear(ctx["model"]["amount_of_input_neurons"], ctx["model"]["amount_of_output_neurons"]))
        
    if ctx["model"]["nonlin_output_func"] == "softmax":
        va_params.append(nn.Softmax(dim=1))
    if ctx["model"]["nonlin_output_func"] == "sigmoid":
        va_params.append(nn.Sigmoid())
        
    ordered_va_params = OrderedDict([(str(i),layer) for i, layer in enumerate(va_params)])
    return nn.Sequential(ordered_va_params)