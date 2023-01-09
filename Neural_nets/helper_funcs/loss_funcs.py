import torch
from torch import nn


def RMSELoss_repaired(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))


def CrossEntropyLoss_repaired(pred, yb):
    yb = torch.argmax(yb, axis=1)
    loss_func = nn.CrossEntropyLoss()
    return loss_func(pred, yb)


# данную функцию потерь можно использовать только для двоичной классификации
def BCELoss_repaired(pred, yb):
    soft = nn.Softmax(dim=1)
    pred = soft(pred)
    loss_func = nn.BCELoss()
    return loss_func(pred, yb)


def KLDivLoss_repaired(pred, yb):
    log_soft = nn.LogSoftmax(dim=1)
    soft = nn.Softmax(dim=1)
    pred = log_soft(pred)
    yb = soft(yb)
    loss_func = nn.KLDivLoss(reduction = 'batchmean')
    return loss_func(pred, yb)


def Get_loss_func(ctx):
    if ctx["model"]["loss_func"] == "RMSELoss":
        return RMSELoss_repaired
    if ctx["model"]["loss_func"] == "CrossEntropyLoss":
        return CrossEntropyLoss_repaired
    if ctx["model"]["loss_func"] == "BCELoss":
        return BCELoss_repaired
    if ctx["model"]["loss_func"] == "KLDivLoss":
        return KLDivLoss_repaired
    
    raise NameError("Wrong loss func")