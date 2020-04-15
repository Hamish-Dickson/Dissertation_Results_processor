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
    w9901_improve = read('C:\\Users\\bktzg\\9901 improves.csv')
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

    w1000_average = read('C:\\Users\\bktzg\\1000 averages.csv')
    w9901_average = read('C:\\Users\\bktzg\\9901 averages.csv')
    w9010_average = read('C:\\Users\\bktzg\\9010 averages.csv')
    w8020_average = read('C:\\Users\\bktzg\\8020 averages.csv')
    w7030_average = read('C:\\Users\\bktzg\\7030 averages.csv')
    w6040_average = read('C:\\Users\\bktzg\\6040 averages.csv')
    w5050_average = read('C:\\Users\\bktzg\\5050 averages.csv')
    w4060_average = read('C:\\Users\\bktzg\\4060 averages.csv')
    w3070_average = read('C:\\Users\\bktzg\\3070 averages.csv')
    w2080_average = read('C:\\Users\\bktzg\\2080 averages.csv')
    w1090_average = read('C:\\Users\\bktzg\\1090 averages.csv')
    w0100_average = read('C:\\Users\\bktzg\\0100 averages.csv')

    surrogateBaseline_improves = read('C:\\Users\\bktzg\\baselineSurrogate improves.csv')
    surrogateBaseline_averages = read('C:\\Users\\bktzg\\baselineSurrogate averages.csv')
    surrogate_improves = read('C:\\Users\\bktzg\\surrogate improves.csv')
    surrogate_averages = read('C:\\Users\\bktzg\\surrogate averages.csv')

    return baseline_improve, twoPoint_improve, multiPoint_improve, aggressiveMutate_improve, baseline_average, \
           twoPoint_average, multiPoint_average, aggressiveMutate_average, w1000_improve, w9901_improve, w9010_improve, \
           w8020_improve, \
           w7030_improve, w6040_improve, w5050_improve, w4060_improve, w3070_improve, w2080_improve, w1090_improve, \
           w0100_improve, w1000_average, w9901_average, w9010_average, w8020_average, w7030_average, w6040_average, \
           w5050_average, \
           w4060_average, w3070_average, w2080_average, w1090_average, w0100_average, surrogateBaseline_improves, \
           surrogate_improves, surrogateBaseline_averages, surrogate_averages


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


def plot_averages(baseline, twopoint, multipoint, aggressive):
    plt.plot(baseline, label='Baseline')
    plt.plot(twopoint, label='Two Point Crossover')
    plt.plot(multipoint, label='Multi Point Mutation')
    plt.plot(aggressive, label='Aggressive Mutation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness score')
    plt.title('Comparison of average score through iterations between\n various GA operators')
    plt.legend()
    plt.autoscale()

    plt.show()


def boxplot(baseline_average_results, twoPoint_average_results, multiPoint_average_results,
            aggressiveMutate_average_results):
    data = [baseline_average_results, twoPoint_average_results,
            multiPoint_average_results, aggressiveMutate_average_results]

    labels = ['Baseline \nStandard Deviation:\n' + str(np.std(baseline_average_results, dtype=np.float64)),
              'Two Point Crossover\nStandard Deviation:\n' + str(np.std(twoPoint_average_results, dtype=np.float64)),
              'Multi Point Mutation\nStandard Deviation:\n' + str(np.std(multiPoint_average_results, dtype=np.float64)),
              'Aggressive Mutation\nStandard Deviation:\n' + str(
                  np.std(aggressiveMutate_average_results, dtype=np.float64))]
    plt.rcParams["font.size"] = "8"
    fig, axs = plt.subplots()

    axs.set_xticklabels(labels)
    plt.rcParams["font.size"] = "6"
    box_plot = axs.boxplot(data, showmeans=True)

    h, l = axs.get_legend_handles_labels()
    h.append(box_plot.get('means')[0])
    h.append(box_plot.get('medians')[0])
    h.append(box_plot.get('fliers')[0])

    l = ['Mean', 'Median', 'Outliers']
    plt.rcParams["legend.loc"] = 'lower right'
    plt.legend(h, l)

    plt.show()


def weighted_plot_all(w1000_results, w9010_results, w8020_results, w7030_results, w6040_results, w5050_results,
                      w4060_results, w3070_results, w2080_results, w1090_results, w0100_results):
    for i in range(0, len(w8020_results)):
        w1000_results[i] = w1000_results[i] / 76 * 100
        w9010_results[i] = w9010_results[i] / 69.1 * 100
        w8020_results[i] = w8020_results[i] / 62.2 * 100
        w7030_results[i] = w7030_results[i] / 55.3 * 100
        w6040_results[i] = w6040_results[i] / 48.4 * 100
        w5050_results[i] = w5050_results[i] / 41.5 * 100
        w4060_results[i] = w4060_results[i] / 34.6 * 100
        w3070_results[i] = w3070_results[i] / 27.7 * 100
        w2080_results[i] = w2080_results[i] / 20.8 * 100
        w1090_results[i] = w1090_results[i] / 13.9 * 100
        w0100_results[i] = w0100_results[i] / 7 * 100

    plt.plot(w1000_results, label='100% evaluation, 0% distance', color='#FE0002', alpha=0.8)
    plt.plot(w9010_results, label='90% evaluation, 10% distance', color='#EC0015', alpha=0.8)
    plt.plot(w8020_results, label='80% evaluation, 20% distance', color='#D80027', alpha=0.8)
    plt.plot(w7030_results, label='70% evaluation, 30% distance', color='#B4003A', alpha=0.8)
    plt.plot(w6040_results, label='60% evaluation, 40% distance', color='#A1015D', alpha=0.8)
    plt.plot(w5050_results, label='50% evaluation, 50% distance', color='#82007D', alpha=0.8)
    plt.plot(w4060_results, label='40% evaluation, 60% distance', color='#63009E', alpha=0.8)
    plt.plot(w3070_results, label='30% evaluation, 70% distance', color='#4700C8', alpha=0.8)
    plt.plot(w2080_results, label='20% evaluation, 80% distance', color='#2A00D5', alpha=0.8)
    plt.plot(w1090_results, label='10% evaluation, 90% distance', color='#1700DD', alpha=0.8)
    plt.plot(w0100_results, label='0% evaluation, 100% distance', color='#0302FC', alpha=0.8)

    plt.xlabel('Generation')
    plt.ylabel('% of Global Maximum Fitness Achieved')
    plt.title('Comparison of % of Global Maximum achievable fitness\n achieved through iterations\n'
              'between various GA operators')

    plt.ylim(45, 93)
    plt.legend(prop={'size': 8})

    plt.show()


def weighted_plot_some(w1000_results, w9010_results, w8020_results, w7030_results, w6040_results, w5050_results,
                       w4060_results):
    plt.plot(w1000_results, label='100% evaluation, 0% distance', color='#FE0002', alpha=0.8)
    plt.plot(w9010_results, label='90% evaluation, 10% distance', color='#EC0015', alpha=0.8)
    plt.plot(w8020_results, label='80% evaluation, 20% distance', color='#D80027', alpha=0.8)
    plt.plot(w7030_results, label='70% evaluation, 30% distance', color='#B4003A', alpha=0.8)
    plt.plot(w6040_results, label='60% evaluation, 40% distance', color='#A1015D', alpha=0.8)
    plt.plot(w5050_results, label='50% evaluation, 50% distance', color='#82007D', alpha=0.8)
    plt.plot(w4060_results, label='40% evaluation, 60% distance', color='#63009E', alpha=0.8)

    plt.xlabel('Generation')
    plt.ylabel('% of Global Maximum Fitness Achieved')
    plt.title('Comparison of % of Global Maximum achievable fitness\n achieved through iterations\n'
              'between various GA operators')

    plt.ylim(71, 77)
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
    plt.grid(axis='y', linestyle='--')
    plt.bar(ypos, vals, align='center', alpha=1)
    plt.xticks(ypos, labels)
    plt.ylabel('% Global Maximum')
    plt.xlabel('Ratio of Automated Evaluation : Entropy value')
    plt.title('Comparison of % of Global Maximum achievable fitness\n achieved through iterations\n'
              'between various GA operators')
    plt.ylim(65, 95)

    plt.show()


def barchart_differences(w1000_average_results, w9901_average_results, w9010_average_results, w8020_average_results,
                         w7030_average_results,
                         w6040_average_results, w5050_average_results, w4060_average_results, w3070_average_results,
                         w2080_average_results, w1090_average_results, w0100_average_results):
    labels = ['100%/0%', '99%/1%', '90%/10%', '80%/20% ', '70%/30%', '60%/40%', '50%/50%', '40%/60%', '30%/70%',
              '20%/80%', '10%/90%', '0%/100%']

    vals = [w1000_average_results[-1] - w1000_average_results[0], w9901_average_results[-1] - w9901_average_results[0],
            w9010_average_results[-1] - w9010_average_results[0],
            w8020_average_results[-1] - w8020_average_results[0], w7030_average_results[-1] - w7030_average_results[0],
            w6040_average_results[-1] - w6040_average_results[0], w5050_average_results[-1] - w5050_average_results[0],
            w4060_average_results[-1] - w4060_average_results[0], w3070_average_results[-1] - w3070_average_results[0],
            w2080_average_results[-1] - w2080_average_results[0], w1090_average_results[-1] - w1090_average_results[0],
            w0100_average_results[-1] - w0100_average_results[0]]

    ypos = np.arange(len(labels))

    print(w9010_average_results[0], w9010_average_results[-1])
    print(vals)
    plt.rcParams["font.size"] = "6"
    plt.grid(axis='y', linestyle='--')
    plt.bar(ypos, vals, align='center', alpha=1)
    plt.xticks(ypos, labels)
    plt.ylabel('Change in average over run')
    plt.xlabel('Ratio of Automated Evaluation : Entropy value')
    plt.title('Comparison of change in average score over run of genetic algorithm between various GA operators')
    plt.autoscale()
    plt.show()


def plot_surrogate(surrogateBaseline_improves, surrogate_improves):
    plt.plot(surrogateBaseline_improves, label='Baseline')
    plt.plot(surrogate_improves, label='Surrogate')
    plt.xlabel('Generation')
    plt.ylabel('Fitness score')
    plt.title('Comparison of maximum score through iterations between\nsurrogate and baseline evaluation')
    plt.legend()
    plt.autoscale()

    plt.show()


def plot_surrogate_average(surrogateBaseline_averages, surrogate_averages):
    plt.plot(surrogateBaseline_averages, label='Baseline')
    plt.plot(surrogate_averages, label='Surrogate')
    plt.xlabel('Generation')
    plt.ylabel('Fitness score')
    plt.title('Comparison of average score through iterations between\nsurrogate and baseline evaluation')
    plt.legend()
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

w1000_improve_results = getVals()[8]
w9901_improve_results = getVals()[9]
w9010_improve_results = getVals()[10]
w8020_improve_results = getVals()[11]
w7030_improve_results = getVals()[12]
w6040_improve_results = getVals()[13]
w5050_improve_results = getVals()[14]
w4060_improve_results = getVals()[15]
w3070_improve_results = getVals()[16]
w2080_improve_results = getVals()[17]
w1090_improve_results = getVals()[18]
w0100_improve_results = getVals()[19]

w1000_average_results = getVals()[20]
w9901_average_results = getVals()[21]
w9010_average_results = getVals()[22]
w8020_average_results = getVals()[23]
w7030_average_results = getVals()[24]
w6040_average_results = getVals()[25]
w5050_average_results = getVals()[26]
w4060_average_results = getVals()[27]
w3070_average_results = getVals()[28]
w2080_average_results = getVals()[29]
w1090_average_results = getVals()[30]
w0100_average_results = getVals()[31]

surrogateBaseline_improves = getVals()[32]
surrogate_improves = getVals()[33]

surrogateBaseline_averages = getVals()[34]
surrogate_averages = getVals()[35]

plot(baseline_improve_results, twoPoint_improve_results, multiPoint_improve_results, aggressiveMutate_improve_results)

plot_averages(baseline_average_results, twoPoint_average_results, multiPoint_average_results,
              aggressiveMutate_average_results)

boxplot(baseline_average_results, twoPoint_average_results, multiPoint_average_results,
        aggressiveMutate_average_results)

weighted_plot_all(w1000_improve_results, w9010_improve_results, w8020_improve_results, w7030_improve_results,
                  w6040_improve_results, w5050_improve_results, w4060_improve_results,
                  w3070_improve_results, w2080_improve_results, w1090_improve_results, w0100_improve_results)

weighted_plot_some(w1000_improve_results, w9010_improve_results, w8020_improve_results, w7030_improve_results,
                   w6040_improve_results, w5050_improve_results, w4060_improve_results)

barchart_percentages(w1000_improve_results, w9010_improve_results, w8020_improve_results, w7030_improve_results,
                     w6040_improve_results, w5050_improve_results,
                     w4060_improve_results,
                     w3070_improve_results, w2080_improve_results, w1090_improve_results, w0100_improve_results)

barchart_differences(w1000_average_results, w9901_average_results, w9010_average_results, w8020_average_results,
                     w7030_average_results,
                     w6040_average_results, w5050_average_results, w4060_average_results, w3070_average_results,
                     w2080_average_results, w1090_average_results, w0100_average_results)

plot_surrogate(surrogateBaseline_improves, surrogate_improves)

plot_surrogate_average(surrogateBaseline_averages, surrogate_averages)
