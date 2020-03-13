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

    w1000_improve = read('C:\\Users\\bktzg\\1000 improves.csv')
    w9010_improve = read('C:\\Users\\bktzg\\9010 improves.csv')
    w8020_improve = read('C:\\Users\\bktzg\\8020 improves.csv')
    w7030_improve = read('C:\\Users\\bktzg\\7030 improves.csv')
    w6040_improve = read('C:\\Users\\bktzg\\6040 improves.csv')
    w5050_improve = read('C:\\Users\\bktzg\\5050 improves.csv')
    w4060_improve = read('C:\\Users\\bktzg\\4060 improves.csv')
    w3070_improve = read('C:\\Users\\bktzg\\3070 improves.csv')
    w2080_improve = read('C:\\Users\\bktzg\\2080 improves.csv')
    w1090_improve = read('C:\\Users\\bktzg\\1090 improves.csv')
    w0100_improve = read('C:\\Users\\bktzg\\0100 improves.csv')

    return baseline_improve, twoPoint_improve, multiPoint_improve, aggressiveMutate_improve, baseline_average, \
           twoPoint_average, multiPoint_average, aggressiveMutate_average, w1000_improve, w9010_improve, w8020_improve, \
           w7030_improve, w6040_improve, w5050_improve, w4060_improve, w3070_improve, w2080_improve, w1090_improve, w0100_improve


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


def weighted_plot(w1000_results, w9010_results, w8020_results, w7030_results, w6040_results, w5050_results,
                  w4060_results, w3070_results, w2080_results, w1090_results, w0100_results):
    for i in range(0, len(w8020_results)):
        w1000_results[i] = w1000_results[i] / 76 * 100
        w9010_results[i] = w9010_results[i] / 74.7 * 100
        w8020_results[i] = w8020_results[i] / 62.2 * 100
        w7030_results[i] = w7030_results[i] / 55.3 * 100
        w6040_results[i] = w6040_results[i] / 48.4 * 100
        w5050_results[i] = w5050_results[i] / 41.5 * 100
        w4060_results[i] = w4060_results[i] / 34.6 * 100
        w3070_results[i] = w3070_results[i] / 27.7 * 100
        w2080_results[i] = w2080_results[i] / 20.8 * 100
        w1090_results[i] = w1090_results[i] / 13.9 * 100
        w0100_results[i] = w0100_results[i] / 7 * 100

    plt.plot(w1000_results, label='100% evaluation, 0% distance')
    plt.plot(w9010_results, label='90% evaluation, 10% distance')
    plt.plot(w8020_results, label='80% evaluation, 20% distance')
    plt.plot(w7030_results, label='70% evaluation, 30% distance')
    plt.plot(w6040_results, label='60% evaluation, 40% distance')
    plt.plot(w5050_results, label='50% evaluation, 50% distance')
    plt.plot(w4060_results, label='40% evaluation, 60% distance')
    plt.plot(w3070_results, label='30% evaluation, 70% distance')
    plt.plot(w2080_results, label='20% evaluation, 80% distance')
    plt.plot(w1090_results, label='10% evaluation, 90% distance')
    plt.plot(w0100_results, label='0% evaluation, 100% distance')

    plt.xlabel('Generation')
    plt.ylabel('% of Global maximum fitness achieved')
    plt.title('Comparison of % of Global Maximum achievable fitness\n achieved through iterations\n'
              'between various GA operators')

    # plt.ylim(70, 80)
    plt.legend()

    plt.show()


def barchart_percentages(w1000_results, w9010_results, w8020_results, w7030_results, w6040_results, w5050_results,
                         w4060_results, w3070_results, w2080_results, w1090_results, w0100_results):
    labels = ['100%/0%', '90%/10%', '80%/20% ', '70%/30%', '60%/40%', '50%/50%', '40%/60%', '30%/70%',
              '20%/80%', '10%/90%', '0%/100%']

    vals = [np.amax(w1000_results), np.amax(w9010_results), np.amax(w8020_results), np.amax(w7030_results),
            np.amax(w6040_results), np.amax(w5050_results), np.amax(w4060_results), np.amax(w3070_results),
            np.amax(w2080_results), np.amax(w1090_results), np.amax(w0100_results)]

    ypos = np.arange(len(labels))

    plt.rcParams["font.size"] = "6"
    plt.bar(ypos, vals, align='center', alpha=0.5)
    plt.xticks(ypos, labels)
    plt.ylabel('% Global Maximum')
    plt.xlabel('Ratio of automated evaluation : Entropy value')
    plt.title('Comparison of % of Global Maximum achievable fitness\n achieved through iterations\n'
              'between various GA operators')
    plt.ylim(65, 90)
    plt.show()


baseline_improve_results = getVals()[0]
twoPoint_improve_results = getVals()[1]
multiPoint_improve_results = getVals()[2]
aggressiveMutate_improve_results = getVals()[3]

baseline_average_results = getVals()[4]
twoPoint_average_results = getVals()[5]
multiPoint_average_results = getVals()[6]
aggressiveMutate_average_results = getVals()[7]

w1000_results = getVals()[8]
w9010_results = getVals()[9]
w8020_results = getVals()[10]
w7030_results = getVals()[11]
w6040_results = getVals()[12]
w5050_results = getVals()[13]
w4060_results = getVals()[14]
w3070_results = getVals()[15]
w2080_results = getVals()[16]
w1090_results = getVals()[17]
w0100_results = getVals()[18]

plot(baseline_improve_results, twoPoint_improve_results, multiPoint_improve_results, aggressiveMutate_improve_results)

boxplot(baseline_average_results, twoPoint_average_results, multiPoint_average_results,
        aggressiveMutate_average_results)

weighted_plot(w1000_results, w9010_results, w8020_results, w7030_results, w6040_results, w5050_results, w4060_results,
              w3070_results, w2080_results, w1090_results, w0100_results)

barchart_percentages(w1000_results, w9010_results, w8020_results, w7030_results, w6040_results, w5050_results,
                     w4060_results,
                     w3070_results, w2080_results, w1090_results, w0100_results)
