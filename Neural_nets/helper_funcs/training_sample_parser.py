import torch
from torch.utils.data import TensorDataset
import csv
from math import floor, ceil
import json


def Parse_training_sample(ctx):
    xx = []
    yy = []
    with open(ctx["training_sample"]) as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        for x, y in reader:
            xx.append(json.loads(x))
            yy.append(json.loads(y))

    ctx["model"].update({"amount_of_input_neurons": len(xx[0])})
    ctx["model"].update({"amount_of_output_neurons": len(yy[0])})

    x = torch.tensor(xx)
    y = torch.tensor(yy)
    # количество примеров в тренировочной выборке
    N = len(yy)

    # 80% выборки трениировочные, 20% - тестовые
    N_train = floor(N*0.8)

    # генерируем удобное представление данных
    test_ds = TensorDataset(x[N_train:], y[N_train:])
    
    if ctx["sampling_training_set_from_unique_values"]:
        #считаем частоту встречаемости выходов для одного входа
        len_y = len(yy[0])
        x_uniq = {x : [0 for _ in range(len_y)] for x in set([str(x) for x in xx[:N_train]])}
        for i, x in enumerate(xx[:N_train]):
            for j, y in enumerate(yy[i]):
                x_uniq[str(x)][j] += y
        
        # добавляем в выборку наиболее часто встречающийся выход
        y_for_x_uniq = [[0.0 for _ in range(len_y)] for __ in range(len(x_uniq.keys()))]
        for i,x in enumerate(x_uniq.values()):
            y_for_x_uniq[i][x.index(max(x))] = 1.0
        
        x_train = torch.tensor([json.loads(x) for x in x_uniq.keys()])
        y_train = torch.tensor(y_for_x_uniq)

        if ctx["expand_unique_training_set_to_initial_length"]:
            x_train = torch.cat([x_train for _ in range(ceil(N_train/len(x_uniq.keys())))])
            y_train = torch.cat([y_train for _ in range(ceil(N_train/len(x_uniq.keys())))])

        return TensorDataset(x_train, y_train), test_ds

    train_ds = TensorDataset(x[:N_train], y[:N_train])
    return train_ds, test_ds