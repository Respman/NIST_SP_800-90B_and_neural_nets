import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):

    def __init__(self, inputSize, inputDim, hiddenNum, outputDim, layerNum, direction):
        super().__init__()
        
        self.input_data_vector_size = inputSize
        self.inputDim = inputDim

        self.cell = nn.RNN(input_size=inputDim, hidden_size=hiddenNum,
                           num_layers=layerNum, bidirectional=direction,
                           dropout=0.0, nonlinearity="tanh", batch_first=True)
        if direction:
            self.fc = nn.Linear(2*hiddenNum, outputDim)
        else:
            self.fc = nn.Linear(hiddenNum, outputDim)

    def forward(self, x):
        # надо поделить выборку на промежутки (то есть "склеить" в один вектор слои, которые отвечают за одну позицию)
        x = torch.tensor([a.reshape(self.input_data_vector_size//self.inputDim,self.inputDim) for a in x.numpy()])
        rnnOutput, hn = self.cell(x,)
        
        # с последнего нейрона берем полученный вектор признаков длины hiddenNum, который
        # "хранит" в себе всю накопленную взвешенную информацию с предыдущих слоев и входов
        fcOutput = self.fc(rnnOutput[:, -1, :].squeeze())
        return F.log_softmax(fcOutput)