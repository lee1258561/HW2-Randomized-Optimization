import matplotlib.pyplot as plt
import numpy as np
import os

def plot_and_save(x, ys, labels, title, x_axis, y_axis, axis_range="auto", ylim=None, fig_path='fig', format='png', loc="best"):
    if axis_range is None:
        plt.axis([x[0], x[-1], 0, 1])
    elif type(axis_range) == type(list()):
        plt.axis(axis_range)
    elif axis_range == 'auto':
        pass

    if ylim is not None:
        plt.ylim(*ylim)

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)

    lines = []
    for y, label in zip(ys, labels):
        l, = plt.plot(x, y, label=label, linewidth=1)
    plt.legend(loc=loc)
    plt.grid(True, linestyle = "-.", color = '0.3')

    plt.savefig(fig_path + '.' + format, format=format, dpi=300)
    plt.clf()

def plot_bar_chart(x, ys, labels, title, x_axis, y_axis, fig_path='fig', format='png'):
    x_index = np.array(list(range(len(x))))
    bar_width = 0.35
    opacity = 0.8

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)

    bars = []
    for i in range(len(ys)):
        rects1 = plt.bar(x_index + i * bar_width, ys[i], bar_width, alpha=opacity, label=labels[i])

    plt.xticks(x_index + float(bar_width) * i / 2, x)
    plt.legend(loc="best")


    plt.tight_layout()
    plt.grid(True, linestyle = "-.", color = '0.3')

    plt.savefig(fig_path + '.' + format, format=format, dpi=300)
    plt.clf()


def create_path(*arg, filename=None):
    path = os.getcwd()
    for directory in arg:
        path = os.path.join(path, directory)
        if not os.path.exists(path):
            print('%s doesn\'t exist, creating...' % path)
            os.mkdir(path)

    if filename:
        path = os.path.join(path, filename)
    return path

def load_log(filename, log_dir="ralog", part2=False):
    error, train_acc, test_acc = [], [], []
    with open(os.path.join(log_dir, filename), "r") as f:
        for line in f:
            log = list(map(float, line.strip().split(',')))
            error.append(log[0])
            if not part2:
                train_acc.append(log[1])
                test_acc.append(log[2])

    return error, train_acc, test_acc

def create_fig(plot_task, legend, title, fig_path, format="png", locs=["best", "best", "best"]):
    errors, train_accs, test_accs, max_dim = [], [], [], 0
    for filename in plot_task:
        error, train_acc, test_acc = load_log(filename)
        errors.append(error)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        max_dim = max(max_dim, len(error))

    for error, train_acc, test_acc in zip(errors, train_accs, test_accs):
        if len(error) < max_dim:
            error += [error[-1]] * (max_dim - len(error))
            train_acc += [train_acc[-1]] * (max_dim - len(train_acc))
            test_acc += [test_acc[-1]] * (max_dim - len(test_acc))

    x = list(range(len(errors[0])))
    error_title = title + " Error"
    plot_and_save(x, errors, legend, error_title, "Iterations", "Error", fig_path=fig_path + "_error", format=format, loc=locs[0])
    train_title = title + " Training Acc"
    plot_and_save(x, train_accs, legend, train_title, "Iterations", "Acc %", ylim=[0, 100], fig_path=fig_path + "_train", format=format, loc=locs[1])
    test_title = title + " Testing Acc"
    plot_and_save(x, test_accs, legend, test_title, "Iterations", "Acc %", ylim=[0, 100], fig_path=fig_path + "_test", format=format, loc=locs[2])

def process_restart(error, train_acc, test_acc):
    p_error, p_train_acc, p_test_acc = [], [], []
    cur_error_i = 0
    for i in range(len(error)):
        if error[i] < error[cur_error_i]:
            cur_error_i = i
        p_error.append(error[cur_error_i])
        p_train_acc.append(train_acc[cur_error_i])
        p_test_acc.append(test_acc[cur_error_i])

    return p_error, p_train_acc, p_test_acc


def main():
    fig_dir = create_path("fig")
    # fig_path = os.path.join(fig_dir, "RHC")
    # error, train_acc, test_acc = load_log("RHC_50000.log")
    # x = list(range(len(error)))
    # plot_and_save(x, [error], ['50000 iterations'], "RHC Error", "Iterations", "Error", fig_path=fig_path + "_error", format="png")
    # plot_and_save(x, [train_acc, test_acc], ['Training Acc', 'Testing Acc'], "RHC Accuracy", "Iterations", "Acc %", fig_path=fig_path + "_accuracy", format="png")

    # error, train_acc, test_acc = load_log("restart10_RHC_50000.log")
    # # error, train_acc, test_acc = process_restart(error, train_acc, test_acc)
    # plot_and_save(list(range(len(train_acc))), 
    #                [train_acc, test_acc], 
    #                ["Training Accuracy", "Testing Accuracy"], 
    #                "Accuracy vs Restart Time", "# Restart", "Acc %", 
    #                fig_path=fig_path + "_restart", format='png')

    # plot_bar_chart(list(range(len(train_acc))), 
    #                [error], 
    #                ["Training Error"], 
    #                "Error vs Restart Time", "# Restart", "Error", 
    #                fig_path=fig_path + "_restart_error", format='png')


    # SA_temp_plots = ["SA_50000_TEMP100000.0_COOL0.995.log",
    #                  "SA_50000_TEMP1.0E8_COOL0.995.log",
    #                  "SA_50000_TEMP1.0E11_COOL0.995.log",
    #                  "SA_50000_TEMP1.0E14_COOL0.995.log",
    #                  "SA_50000_TEMP1.0E17_COOL0.995.log"]
    # legend = ["temp = 1e5", "temp = 1e8", "temp = 1e11", "temp = 1e14", "temp = 1e17"]

    # fig_path = os.path.join(fig_dir, "SA_temp")
    # create_fig(SA_temp_plots, legend, "SA Temperture", fig_path)

    # SA_cool_plots = ["SA_50000_TEMP1.0E11_COOL0.9.log",
    #                  "SA_50000_TEMP1.0E11_COOL0.95.log",
    #                  "SA_50000_TEMP1.0E11_COOL0.99.log",
    #                  "SA_50000_TEMP1.0E11_COOL0.995.log",
    #                  "SA_50000_TEMP1.0E11_COOL0.999.log",]
    # legend = ["cool down = 0.9", "cool down = 0.95", "cool down = 0.99", "cool down = 0.995", "cool down = 0.999"]

    # fig_path = os.path.join(fig_dir, "SA_cool_down")
    # create_fig(SA_cool_plots, legend, "SA Cool Down", fig_path, locs=["upper right", "lower right", "lower right"])

    # GA_pop_plots = ["GA_5000_POP200_MAT100_MU10.log",
    #                 "GA_5000_POP500_MAT100_MU10.log",
    #                 "GA_5000_POP1000_MAT100_MU10.log",]
    # legend = ["population = 200", "population = 500", "population = 1000"]

    # fig_path = os.path.join(fig_dir, "GA_pop")
    # create_fig(GA_pop_plots, legend, "GA Population", fig_path)

    # GA_mate_plots = ["GA_5000_POP500_MAT50_MU10.log",
    #                  "GA_5000_POP500_MAT75_MU10.log",
    #                  "GA_5000_POP500_MAT100_MU10.log",
    #                  "GA_5000_POP500_MAT125_MU10.log",
    #                  "GA_5000_POP500_MAT150_MU10.log"]
    # legend = ["mate = 50", "mate = 75", "mate = 100", "mate = 125", "mate = 150"]

    # fig_path = os.path.join(fig_dir, "GA_mate")
    # create_fig(GA_mate_plots, legend, "GA Mate", fig_path)

    # GA_mutate_plots = ["GA_5000_POP500_MAT75_MU50.log",
    #                  "GA_5000_POP500_MAT75_MU75.log",
    #                  "GA_5000_POP500_MAT75_MU100.log",
    #                  "GA_5000_POP500_MAT75_MU125.log",
    #                  "GA_5000_POP500_MAT75_MU150.log"]
    # legend = ["mutate = 50", "mutate = 75", "mutate = 100", "mutate = 125", "mutate = 150"]

    # fig_path = os.path.join(fig_dir, "GA_mutate")
    # create_fig(GA_mutate_plots, legend, "GA Mutate", fig_path)

    # Compare_plots = ["RHC_50000.log",
    #                  "SA_50000_TEMP1.0E11_COOL0.95.log",
    #                  "GA_5000_POP500_MAT75_MU150.log"]

    # legend = ["RHC", "SA", "GA"]

    # fig_path = os.path.join(fig_dir, "Algorithm Comparison")
    # create_fig(Compare_plots, legend, "Compare", fig_path)

    # Promblem 1 TSP
    # RHC_fitness, _, _ = load_log("TSP_RHC.log", part2=True);
    # SA_fitness, _, _ = load_log("TSP_SA.log", part2=True);
    # GA_fitness, _, _ = load_log("TSP_GA.log", part2=True);
    # MIMIC_fitness, _, _ = load_log("TSP_MIMIC.log", part2=True);
    # fitnesses = [RHC_fitness, SA_fitness, GA_fitness, MIMIC_fitness]
    # legend = ["RHC", "SA", "GA", "MIMIC"]
    # fig_path = os.path.join(fig_dir, "TSP_fitness")
    # plot_and_save(list(range(len(fitnesses[0]))), fitnesses, legend, "Traveling Salesman", "Iterations", "Fitness", fig_path=fig_path, format="png")


    # RHC_fitness, _, _ = load_log("FP_RHC.log", part2=True);
    # SA_fitness, _, _ = load_log("FP_SA.log", part2=True);
    # GA_fitness, _, _ = load_log("FP_GA.log", part2=True);
    # MIMIC_fitness, _, _ = load_log("FP_MIMIC.log", part2=True);
    # fitnesses = [RHC_fitness, SA_fitness, GA_fitness, MIMIC_fitness + [MIMIC_fitness[-1]] * 10000]
    # legend = ["RHC", "SA", "GA", "MIMIC"]
    # fig_path = os.path.join(fig_dir, "FP_fitness")
    # plot_and_save(list(range(len(fitnesses[0]))), fitnesses, legend, "Four Peaks", "Iterations", "Fitness", fig_path=fig_path, format="png")

    # RHC_fitness, _, _ = load_log("MAXK_RHC.log", part2=True);
    # SA_fitness, _, _ = load_log("MAXK_SA.log", part2=True);
    GA_fitness, _, _ = load_log("MAXK_GA.log", part2=True);
    MIMIC_fitness, _, _ = load_log("MAXK_MIMIC.log", part2=True);
    fitnesses = [GA_fitness, MIMIC_fitness + [MIMIC_fitness[int(-1)]] * 45]
    legend = ["GA", "MIMIC"]
    fig_path = os.path.join(fig_dir, "MAXK_fitness")
    plot_and_save(list(range(len(fitnesses[0]))), fitnesses, legend, "Max-K-Coloring", "Iterations", "Fitness", fig_path=fig_path, format="png")


if __name__ == "__main__":
    main()
