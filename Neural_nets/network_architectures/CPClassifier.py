import simpsom as sps
from simpsom.plots import scatter_on_map
import numpy as np
import random
from math import log2


class CPClassifier():
    """
    Реализация архитектуры сети встречного распространения
    
    """
    
    def __init__(self):
        self.kohonen_net = None
        self.grossberg_layer = None


    def _prepare_datasets(self, ds):
        x_ds = np.array([[1 if i == 1 else 0 for i in x] for x in ds.tensors[0]])
        y_ds = np.array([int(log2(int("".join(str(int(i)) for i in y),2))) for y in ds.tensors[1]])
        return x_ds, y_ds


    def test(self, test_ds):
        x_test, y_test = self._prepare_datasets(test_ds)
        counter = 0
        for x,y in zip(np.array(self.kohonen_net.find_bmu_ix(x_test)),y_test):
            if self.grossberg_layer[x] == y:
                counter += 1
        return counter/len(y_test) 


    # проекция тренировочной выборки на карту Кохонена
    def _net_projection(self, ctx, x_train, y_train):
        projection = self.kohonen_net.project_onto_map(x_train, file_name = ctx["file_to_save_projected_data"])

        # если мы хотим вывести только нейроны, которые перейдут в "0"
        #scatter_on_map([projection[y_train==0]],
        #               [[node.pos[0], node.pos[1]] for node in self.kohonen_net.nodes_list],
        #               self.kohonen_net.polygons, color_val=None, show=True, print_out=True,
        #               file_name = ctx["file_to_save_net_projection"])

        scatter_on_map([projection[y_train==i] for i in range(ctx["model"]["amount_of_output_neurons"])],
                       [[node.pos[0], node.pos[1]] for node in self.kohonen_net.nodes_list],
                       self.kohonen_net.polygons, color_val=None, show=True, print_out=True,
                       file_name = ctx["file_to_save_net_projection"])


    # соединаяем полученные классы и выходные значения
    def _train_Grossberg_layer(self, ctx, x_train, y_train):
        grossberg_layer = [[] for _ in range(ctx["model"]["x_dim"]*ctx["model"]["y_dim"])]
        for x,y in zip(self.kohonen_net.find_bmu_ix(x_train),y_train):
            grossberg_layer[x].append(y)
        for i,x in enumerate(grossberg_layer):
            if len(x) == 0:
                grossberg_layer[i].append(random.randrange(ctx["model"]["amount_of_output_neurons"]))

        # определяем выход методом большинства
        self.grossberg_layer = [max(set(y), key=y.count) for y in grossberg_layer]


    def train(self, ctx, train_ds):
        x_train, y_train = self._prepare_datasets(train_ds)

        if ctx["model"]["creating_method"] == "train":
            self.kohonen_net = sps.SOMNet(ctx["model"]["x_dim"], ctx["model"]["y_dim"], x_train, topology='hexagonal',
                                          PBC=True, init='PCA', metric=ctx["model"]["metric"],
                                          neighborhood_fun='gaussian', random_seed=random.randrange(2**32),
                                          GPU=False)
            self.kohonen_net.train(train_algo='batch', start_learning_rate=ctx["model"]["learning_rate"],
                                   epochs=ctx["epochs"], batch_size=ctx["basket_size"])
            self.kohonen_net.save_map(ctx["model"]["file_with_model"])
        elif ctx["model"]["creating_method"] == "load":
            self.kohonen_net = sps.SOMNet(ctx["model"]["x_dim"], ctx["model"]["y_dim"], x_train, topology='hexagonal',
                                          PBC=True, init='PCA', metric=ctx["model"]["metric"],
                                          neighborhood_fun='gaussian', random_seed=random.randrange(2**32),
                                          GPU=False, load_file=ctx["model"]["file_with_model"])
        else: raise NameError("Wrong SOM creating method")

        if ctx["plot_map_by_difference"]:
            self.kohonen_net.plot_map_by_difference(show=True, print_out=True, file_name = ctx["file_to_save_diff_map"])
        if ctx["net_projection"]:
            self._net_projection(ctx, x_train, y_train)

        self._train_Grossberg_layer(ctx, x_train, y_train)
