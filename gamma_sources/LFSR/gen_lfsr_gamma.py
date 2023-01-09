from pylfsr import LFSR
import random
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


def Gen_gamma(poly, state):
    start_pos = 1000
    N = 10**6+poly[0]+1
    L = LFSR(fpoly=poly, initstate = state)
    L.runKCycle(start_pos+N)
    return L.seq[start_pos:]


def main():
    filename = "gamma_48"
    
    state_48 = [random.randint(0,1) for _ in range(48)]
    poly_48 = [48, 47, 44, 42, 41, 37, 36, 35, 31, 30, 27, 24, 23, 21, 20, 17, 16, 15, 13, 10, 9, 8, 7, 4, 3, 1]
    g_48 = Gen_gamma(poly_48, state_48)
    Dump_gamma(g_48, filename, 48)


if __name__ == "__main__":
    main()