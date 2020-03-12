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

    w8020_improve = read('C:\\Users\\bktzg\\8020 improves.csv')

    w9010_improve = read('C:\\Users\\bktzg\\9010 improves.csv')

    w7030_improve = read('C:\\Users\\bktzg\\7030 improves.csv')

    w6040_improve = read('C:\\Users\\bktzg\\6040 improves.csv')

    return baseline_improve, twoPoint_improve, multiPoint_improve, aggressiveMutate_improve, baseline_average, \
           twoPoint_average, multiPoint_average, aggressiveMutate_average, w8020_improve, w9010_improve, \
           w7030_improve, w6040_improve


def plot(baseline, twopoint, multipoint, aggressive):
    plt.plot(baseline, label='Baseline')
    plt.plot(twopoint, label='Two Point Crossover')
    plt.plot(multipoint, label='Multi Point Mutation')
    plt.plot(aggressive, label='Aggressive Mutation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness score')
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
    axs[0, 0].set(ylim=(35, 50))

    axs[0, 1].boxplot(twoPoint_average_results)
    axs[0, 1].set_title(
        'Two Point Crossover\nStandard Deviation:\n' + str(np.std(twoPoint_average_results, dtype=np.float64)))
    axs[0, 1].set_ylabel('Fitness score')
    axs[0, 1].set(ylim=(35, 50))

    axs[1, 0].boxplot(multiPoint_average_results)
    axs[1, 0].set_title(
        'Multi Point Mutation\nStandard Deviation:\n' + str(np.std(multiPoint_average_results, dtype=np.float64)))
    axs[1, 0].set_ylabel('Fitness score')
    axs[1, 0].set(ylim=(35, 50))

    axs[1, 1].boxplot(aggressiveMutate_average_results)
    axs[1, 1].set_title(
        'Aggressive Mutation\nStandard Deviation:\n' + str(np.std(aggressiveMutate_average_results, dtype=np.float64)))
    axs[1, 1].set_ylabel('Fitness score')
    axs[1, 1].set(ylim=(35, 50))

    # plt.autoscale()

    plt.show()


def weighted_plot(w8020_results, w9010_results, w7030_results, w6040_results):
    for i in range(0, len(w8020_results)):
        w8020_results[i] = w8020_results[i] / 104.2 * 100
        w9010_results[i] = w9010_results[i] / 90.1 * 100
        w7030_results[i] = w7030_results[i] / 118.3 * 100
        w6040_results[i] = w6040_results[i] / 132.4 * 100

    plt.plot(w9010_results, label='90% evaluation, 10% distance')
    plt.plot(w8020_results, label='80% evaluation, 20% distance')
    plt.plot(w7030_results, label='70% evaluation, 30% distance')
    plt.plot(w6040_results, label='60% evaluation, 40% distance')
    plt.xlabel('Generation')
    plt.ylabel('% of Global maximum fitness achieved')
    plt.title('Comparison of % of Global Maximum achievable fitness\n achieved through iterations\n'
              'between various GA operators')

    plt.ylim(25,75)
    plt.legend()

    plt.show()


baseline_improve_results = getVals()[0]
twoPoint_improve_results = getVals()[1]
multiPoint_improve_results = getVals()[2]
aggressiveMutate_improve_results = getVals()[3]

baseline_average_results = getVals()[4]
twoPoint_average_results = getVals()[5]
multiPoint_average_results = getVals()[6]
aggressiveMutate_average_results = getVals()[7]

w8020_results = getVals()[8]
w9010_results = getVals()[9]
w7030_results = getVals()[10]
w6040_results = getVals()[11]

plot(baseline_improve_results, twoPoint_improve_results, multiPoint_improve_results, aggressiveMutate_improve_results)

boxplot(baseline_average_results, twoPoint_average_results, multiPoint_average_results,
        aggressiveMutate_average_results)

weighted_plot(w8020_results, w9010_results, w7030_results, w6040_results)
