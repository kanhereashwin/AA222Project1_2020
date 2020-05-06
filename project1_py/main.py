import numpy as np
from helpers import Simple1, Simple2, Simple3, Test
import project1
import plotting

test_problem = Simple1()

optim_history = project1.optimize_history(test_problem.f, test_problem.g, test_problem.x0(), test_problem.n, test_problem.count, test_problem.prob, debug=False)

plotting.plot_rosenbrock()
#plotting.convergence_plot()