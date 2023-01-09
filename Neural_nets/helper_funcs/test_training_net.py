import torch
from torch.utils.data import DataLoader


def Test_training_net(ctx, model, test_ds):
    if ctx["model"]["type"] == "CPClassifier":
        return model.test(test_ds)
    
    test_dl = DataLoader(test_ds, batch_size=ctx["basket_size"])
    relation = 0
    for x,y in test_dl:
        z = model(x)
        _, z_vals = torch.max(z, 1)
        _, y_vals = torch.max(y, 1)
        relation += (z_vals == y_vals).sum().item()

    relation /= len(test_dl)*ctx["basket_size"]
    return relation