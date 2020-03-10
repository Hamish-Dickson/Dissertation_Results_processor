import csv
import matplotlib.pyplot as plt
import numpy as np


def read(filename):
    with open(filename, mode='r') as f:
        vals = []
        reader = csv.reader(f)
        for row in reader:
            for col in row:
                vals.append(col)

        vals.__delitem__(-1)
        return np.array(vals, dtype=np.float32)


def getVals():
    baseline_improve = read('C:\\Users\\bktzg\\baseline improves.csv')
    baseline_average = read('C:\\Users\\bktzg\\baseline averages.csv')

    '''twoPoint_improve = read('C:\\Users\\bktzg\\twopoint improves.csv')
    twoPoint_average = read('C:\\Users\\bktzg\\twopoint averages.csv')
'''
    multiPoint_improve = read('C:\\Users\\bktzg\\multipoint improves.csv')
    multiPoint_average = read('C:\\Users\\bktzg\\multipoint averages.csv')

    aggressiveMutate_improve = read('C:\\Users\\bktzg\\aggressive improves.csv')
    aggressiveMutate_average = read('C:\\Users\\bktzg\\aggressive averages.csv')

    return baseline_improve, multiPoint_improve, aggressiveMutate_improve, baseline_average, multiPoint_average,\
           aggressiveMutate_average


def plot(baseline, multipoint, aggressive):
    plt.plot(baseline, label='baseline')
    plt.plot(multipoint, label='multipoint mutator')
    plt.plot(aggressive, label='aggressive mutator')
    plt.xlabel('Iteration')
    plt.ylabel('Maximum Score')
    plt.title('Comparison of maximum score through iterations between\n various GA operators')
    plt.legend()
    plt.autoscale()

    plt.show()


baseline_improve_results = getVals()[0]
multiPoint_improve_results = getVals()[1]
aggressiveMutate_improve_results = getVals()[2]

plot(baseline_improve_results, multiPoint_improve_results, aggressiveMutate_improve_results)
