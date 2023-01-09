#! /usr/bin/env python3
import csv
# source: https://github.com/hnj2/sp800_90b
import sp800_90b


def main():
    # получаем выборку, с которой будем работать
    gamma = []
    with open("..\gamma_sources\Markov\MultiMMC.csv") as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        for row in reader:
            gamma = bytes([int(r) for r in row])
    gammad = sp800_90b.Data(gamma, 1)
    entropy = gammad.h_multi_markov()
    print(f"entropy: {entropy}")

if __name__ == '__main__':
    main()