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
        return np.array(vals, dtype=np.float64)


def getVals():
    baseline_improve = read('C:\\Users\\bktzg\\baseline improves.csv')
    baseline_average = read('C:\\Users\\bktzg\\baseline averages.csv')

    twoPoint_improve = read('C:\\Users\\bktzg\\twopoint improves.csv')
    twoPoint_average = read('C:\\Users\\bktzg\\twopoint averages.csv')

    multiPoint_improve = read('C:\\Users\\bktzg\\multipoint improves.csv')
    multiPoint_average = read('C:\\Users\\bktzg\\multipoint averages.csv')

    aggressiveMutate_improve = read('C:\\Users\\bktzg\\aggressive improves.csv')
    aggressiveMutate_average = read('C:\\Users\\bktzg\\aggressive averages.csv')

    return baseline_improve, twoPoint_improve, multiPoint_improve, aggressiveMutate_improve, baseline_average, \
           twoPoint_average, multiPoint_average, aggressiveMutate_average


def plot(baseline, twopoint, multipoint, aggressive):
    plt.plot(baseline, label='baseline')
    plt.plot(twopoint, label='two-point crossover')
    plt.plot(multipoint, label='multi-point mutator')
    plt.plot(aggressive, label='aggressive mutator')
    plt.xlabel('Generation')
    plt.ylabel('Maximum Score')
    plt.title('Comparison of maximum score through iterations between\n various GA operators')
    plt.legend()
    plt.autoscale()

    plt.show()


def boxplot(baseline_average_results, twoPoint_average_results, multiPoint_average_results,
            aggressiveMutate_average_results):
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].boxplot(baseline_average_results)
    axs[0, 0].set_title('Baseline \nStandard Deviation:\n' + str(np.std(baseline_average_results, dtype=np.float64)))
    axs[0, 0].set_ylabel('Fitness score')

    axs[0, 1].boxplot(twoPoint_average_results)
    axs[0, 1].set_title(
        'Two point crossover\nStandard Deviation:\n' + str(np.std(twoPoint_average_results, dtype=np.float64)))
    axs[0, 1].set_ylabel('Fitness score')

    axs[1, 0].boxplot(multiPoint_average_results)
    axs[1, 0].set_title(
        'Multi point mutation\nStandard Deviation:\n' + str(np.std(multiPoint_average_results, dtype=np.float64)))
    axs[1, 0].set_ylabel('Fitness score')

    axs[1, 1].boxplot(aggressiveMutate_average_results)
    axs[1, 1].set_title(
        'Aggressive mutation\nStandard Deviation:\n' + str(np.std(aggressiveMutate_average_results, dtype=np.float64)))
    axs[1, 1].set_ylabel('Fitness score')

    plt.autoscale()

    plt.show()


baseline_improve_results = getVals()[0]
twoPoint_improve_results = getVals()[1]
multiPoint_improve_results = getVals()[2]
aggressiveMutate_improve_results = getVals()[3]

baseline_average_results = getVals()[4]
twoPoint_average_results = getVals()[5]
multiPoint_average_results = getVals()[6]
aggressiveMutate_average_results = getVals()[7]

plot(baseline_improve_results, twoPoint_improve_results, multiPoint_improve_results, aggressiveMutate_improve_results)

boxplot(baseline_average_results, twoPoint_average_results, multiPoint_average_results,
        aggressiveMutate_average_results)
