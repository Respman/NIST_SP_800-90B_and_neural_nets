from math import log2
from sys import argv

from helper_funcs.config_parser import Parse_config
from helper_funcs.neural_net_creator import Create_neural_net
from helper_funcs.training_sample_parser import Parse_training_sample
from helper_funcs.test_training_net import Test_training_net
from helper_funcs.train_net import Train_net


def main():
    if len(argv) != 2:
        raise NameError("Usage: python ./neural_network.py config.json")

    ctx = Parse_config(argv[1])
    train_ds, test_ds = Parse_training_sample(ctx)
    model = Create_neural_net(ctx)
    Train_net(ctx, model, train_ds, test_ds)

    print(f"training set relation: {Test_training_net(ctx, model, train_ds)}")
    relation = Test_training_net(ctx, model, test_ds)
    print(f"test set relation: {relation}")
    entropy = -log2(relation)
    print(f"entropy: {entropy}")


if __name__ == '__main__':
    main()