from random import SystemRandom
import csv


def Dump_gamma(gamma, filename, depth):
    with open(f"{filename}.csv", "wt") as fp:
        writer = csv.writer(fp, delimiter=",", lineterminator='\n')
        for i in range(len(gamma)-depth-1):
            sample = gamma[i:i+depth]
            input_vals = [-1.0]*2*depth
            for j in range(depth):
                input_vals[2*j+sample[j]] = 1.0
            output_vals = [0.0]*2
            output_vals[gamma[i+depth+1]] = 1.0
            writer.writerow([input_vals,output_vals])


def main():
    filename = "gamma_20"
    len_gamma = 20
    
    cryptogen = SystemRandom()
    gamma = [cryptogen.randrange(2) for i in range(len_gamma)]
    period_gamma = gamma*(10**6//len_gamma)
    Dump_gamma(period_gamma, filename, len_gamma)


if __name__ == '__main__':
    main()