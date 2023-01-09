from torch import optim
from torch.utils.data import DataLoader
from math import ceil

from helper_funcs.loss_funcs import Get_loss_func
from helper_funcs.learning_rate import Get_learning_rate_sheduler
from helper_funcs.test_training_net import Test_training_net
from helper_funcs.plot import Show_loss_plot, Show_validation_accuracy_plot


def Train_net(ctx, model, train_ds, test_ds):
    if ctx["model"]["type"] == "CPClassifier":
        model.train(ctx, train_ds)
        return

    loss_fn = Get_loss_func(ctx)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.3, nesterov=True)
    scheduler = Get_learning_rate_sheduler(ctx,opt)
    training_results = {'training_loss': [], 'validation_accuracy': []}  

    for epoch in range(ctx["epochs"]):
        if epoch%(ceil(ctx["epochs"]/20)) == 0:
            print(f"epoch: {epoch}")
        
        for xb, yb in DataLoader(train_ds, batch_size=ctx["basket_size"], shuffle=True):  
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if ctx["show_loss_plot"]: training_results['training_loss'].append(loss.data.item())
        
        scheduler.step()
        if ctx["show_validation_accuracy_plot"]:
            test_res = Test_training_net(ctx, model, test_ds)
            training_results['validation_accuracy'].append(test_res)
        
    if ctx["show_loss_plot"]: Show_loss_plot(training_results)
    if ctx["show_validation_accuracy_plot"]: Show_validation_accuracy_plot(training_results)