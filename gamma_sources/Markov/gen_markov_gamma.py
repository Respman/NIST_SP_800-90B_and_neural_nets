import random
import numpy as np
import csv
from math import log2


def Get_probability_table(alph_len, p):
    # перекосим вероятность
    other_p = (1-p)/(alph_len-1)
    prob = [other_p]*alph_len
    prob[random.randint(0,alph_len-1)] = p
    return prob


def main():
    # глубина зависимости в марковской цепи
    depth = 10
    # размер сгенерированной выборки
    amount = 10**6
    # величина "перекошенной" вероятности
    p = 0.75
    # матрица для двоичной марковской цепи
    M = [Get_probability_table(2, p) for _ in range(2**depth)]
    #print(M)
    
    gamma = [random.getrandbits(1) for _ in range(depth)]
    for _ in range(amount+1):
        row_idx = int("".join([str(g) for g in gamma[-depth:]]),2)
        gamma.append(np.random.choice([i for i in range(2)], 1, p=M[row_idx])[0])
    #print(gamma)

    # делим полученную гамму на выборку для нейронной сети
    # и складываем её в csv-файлик
    with open("neuro.csv", "wt") as fp:
        writer = csv.writer(fp, delimiter=",", lineterminator='\n')
        for i in range(len(gamma)-depth-1):
            sample = gamma[i:i+depth]
            input_vals = [-1.0]*2*depth
            for j in range(depth):
                input_vals[2*j+sample[j]] = 1.0
            output_vals = [0.0]*2
            output_vals[gamma[i+depth+1]] = 1.0
            writer.writerow([input_vals,output_vals])
    
    # подготавливаем гамму для теста MultiMMC
    with open("MultiMMC.csv", "wt") as fp:
        writer = csv.writer(fp, delimiter=",")
        writer.writerow(gamma)

    # нужно посчитать истинную мин.энтропию источника
    print(f"H_min per symb: {-log2(p)}")


if __name__ == '__main__':
    main()