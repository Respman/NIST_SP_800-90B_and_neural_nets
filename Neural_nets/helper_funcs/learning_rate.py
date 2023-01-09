from torch import optim
from math import floor

def Get_learning_rate_sheduler(ctx, optimizer):
    if ctx["learning_rate"]["type"] == "MultiStepLR":
        return optim.lr_scheduler.MultiStepLR(optimizer, 
                                              milestones=[floor(ctx["epochs"]/3),floor(ctx["epochs"]*2/3)], 
                                              gamma=0.01)

    raise NameError("Wrong learning rate type")