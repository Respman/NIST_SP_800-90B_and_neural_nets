from network_architectures.FNN import FNN
from network_architectures.ResNet import ResNet
from network_architectures.CPClassifier import CPClassifier
from network_architectures.RNN import RNNModel


def Create_neural_net(ctx):
    if ctx["model"]["type"] == "FNN":
        return FNN(ctx)
    
    if ctx["model"]["type"] == "ResNet":
        return ResNet(ctx["model"]["amount_of_input_neurons"],
                      ctx["model"]["amount_of_output_neurons"],
                      ctx["model"]["amount_hidden_layers"])

    if ctx["model"]["type"] == "CPClassifier":
        return CPClassifier()

    if ctx["model"]["type"] == "RNN":
        return RNNModel(inputSize=ctx["model"]["amount_of_input_neurons"],
                        inputDim=ctx["model"]["amount_of_output_neurons"],
                        hiddenNum=ctx["model"]["amount_of_hidden_neurons"],
                        outputDim=ctx["model"]["amount_of_output_neurons"],
                        layerNum=ctx["model"]["amount_of_recurrent_layers"],
                        direction=ctx["model"]["bidirectional"])

    raise NameError("Wrong model type")