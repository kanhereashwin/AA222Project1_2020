import numpy as np
from matplotlib import pyplot as plt
from helpers import Simple1, Simple2, Simple3
from project1 import optimize_history

def plot_rosenbrock():
    problem = Simple1()
    x_hist_1, _ = optimize_history(problem.f, problem.g, [0,-1], problem.n, problem.count, problem.prob, debug=False)
    problem = Simple1()
    x_hist_2, _ = optimize_history(problem.f, problem.g, [-1, -1], problem.n, problem.count, problem.prob, debug=False)
    problem = Simple1()
    x_hist_3, _ = optimize_history(problem.f, problem.g, [1, -2], problem.n, problem.count, problem.prob, debug=False)
    problem.nolimit()
    x0_list = np.linspace(-3, 3, 100)
    x1_list = np.linspace(-3, 3, 100)
    X0, X1 = np.meshgrid(x0_list, x1_list)
    Z = rosenbrock(X0, X1)
    x_hist_1 = np.array(x_hist_1)
    x_hist_2 = np.array(x_hist_2)
    x_hist_3 = np.array(x_hist_3)
    plt.figure()
    plt.contour(X0, X1, Z, 100)
    plt.plot(x_hist_1[:, 0], x_hist_1[:, 1], '--k')
    plt.plot(x_hist_2[:, 0], x_hist_2[:, 1], '-.k')
    plt.plot(x_hist_3[:, 0], x_hist_3[:, 1], '-k')
    plt.show()
    return None


def convergence_plot():
    for p_type in [Simple1, Simple2, Simple3]:
        problem = p_type()
        init_x = problem.x0()
        x_hist, _ = optimize_history(problem.f, problem.g, init_x, problem.n, problem.count, problem.prob, debug=False)
        problem.nolimit()
        obj_vals = np.empty(len(x_hist))
        print(len(x_hist))
        for pt_idx, pt in enumerate(x_hist):
            obj_vals[pt_idx] = problem.f(pt)
        plt.figure()
        plt.plot(obj_vals, label='Objective value during optimization')
        plt.plot(np.zeros_like(obj_vals), label='Optimal objective value')
        plt.title('Convergence plot for ' + problem.prob + ' with initial point ' + str(init_x))
        plt.xlabel('Number of gradient computations')
        plt.ylabel('f(x)')
        plt.legend()
    plt.show()


def rosenbrock(X0, X1):
    return 100*(X1-X0**2)**2 + (1-X0)**2