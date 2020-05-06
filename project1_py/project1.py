#
# File: project1.py
#

## top-level submission file

'''
Note: Do not import any other modules here.
        To import from another file xyz.py here, type
        import project1_py.xyz
        However, do not import any modules except numpy in those files.
        It's ok to import modules only in files that are
        not imported here (e.g. for your plotting code).
'''
import numpy as np


def optimize(f, g, x0, n, count, prob):
    """
    Args:
        f (function): Function to be optimized
        g (function): Gradient function for `f`
        x0 (np.array): Initial position to start from
        n (int): Number of evaluations allowed. Remember `g` costs twice of `f`
        count (function): takes no arguments are returns current count
        prob (str): Name of the problem. So you can use a different strategy
                 for each problem. `prob` can be `simple1`,`simple2`,`simple3`,
                 `secret1` or `secret2`
    Returns:
        x_best (np.array): best selection of variables found
    """
    _, x_best = optimize_history(f, g, x0, n, count, prob)

    return x_best


class GradientDescent:
    def __init__(self, problem, alpha=0.001):
        self.alpha = alpha
    
    def step(self, x, f_val = None, g_val = None):
        return x - self.alpha*g_val/np.linalg.norm(g_val)


class Momentum():
    def __init__(self, problem, beta = 0.9, alpha=0.001, threshold=1):
        self.beta = beta
        self.alpha = alpha
        self.prev_v = 0
        self.tau = threshold

    def step(self, x, f_val=None, g_val=None):
        g_val = clip_grad(g_val, self.tau)
        self.prev_v = self.beta*self.prev_v - self.alpha*g_val
        
        return x + self.prev_v


class Adagrad():
    def __init__(self, problem, alpha=0.001, epsilon=1e-8, threshold=1):
        self.alpha = alpha
        self.epsilon = epsilon
        self.tau = threshold
        self.xdim = find_xdim(problem)
        self.s = np.zeros(self.xdim)

    def step(self, x, f_val=None, g_val=None):
        g_val = clip_grad(g_val, self.tau)
        new_x = np.copy(x)
        self.s += g_val**2
        new_x -= self.alpha/(self.epsilon + np.sqrt(self.s))*g_val
        #for idx in range(np.size(x)):
        #    new_x[idx] -= self.alpha/(self.epsilon + np.sqrt(self.s[idx]))*g_val[idx] 
        return new_x 

class Adam():
    def __init__(self, problem, alpha=0.001, epsilon = 1e-8, gamma_v = 0.9, gamma_s = 0.999, threshold = 1):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma_v = gamma_v
        self.gamma_s = gamma_s
        self.tau = threshold
        self.t = 0
        self.xdim = find_xdim(problem)
        self.v = np.zeros(self.xdim)
        self.s = np.zeros(self.xdim)

    def step(self, x, f_val=None, g_val=None):
        self.t += 1
        g_val = clip_grad(g_val, self.tau)
        self.v = self.gamma_v*self.v + (1 - self.gamma_v)*g_val
        self.s = self.gamma_s*self.s + (1 - self.gamma_s)*g_val**2
        v_hat = self.v/(1 - self.gamma_v**self.t)
        s_hat = self.s/(1 - self.gamma_s**self.t)
        return x - self.alpha*v_hat / (self.epsilon + np.sqrt(s_hat))


def find_xdim(problem):
    if problem == 'simple1':
        xdim = 2
    elif problem == 'simple2':
        xdim = 2
    elif problem == 'simple3':
        xdim = 4
    elif problem == 'secret1':
        xdim = 50
    elif problem == 'secret2':
        xdim = 1
    elif problem == 'test':
            xdim = 2
    else:
        ValueError('Incorrect problem specified')
    return xdim


def clip_grad(g_val, tau):
    if np.linalg.norm(g_val) > tau:
        g_val = tau * g_val/np.linalg.norm(g_val)
    return g_val

def optimize_history(f, g, old_x, n, count, prob, debug=False):
    x_hist = []
    #optim = Momentum(prob, alpha = 0.001, beta=0.9 )
    #optim = Adam(prob, alpha=0.2, gamma_v=0.9, gamma_s=0.99, threshold=1)
    optim = Adam(prob, alpha=0.2, gamma_v=0.85, gamma_s=0.99, threshold=1)
    if debug:
        print('Solving problem ', prob)
        print('Initial condition is', old_x)
        print('Initial function evaluation', f(old_x))
        print('Initial count ', count())
        print('Number of counts allowed', n)
    x_hist.append(old_x)
    while count() < n:
            g_eval = g(old_x)
            new_x = optim.step(old_x, g_val = g_eval)
            x_hist.append(new_x)
            old_x = np.copy(new_x)
            if debug:
                if count() > 200:
                    break
    if debug:
        print('Final count number ', count())
        print('Final function evaluation ', f(x_hist[-1]))
        print('Final x is ', x_hist[-1])
        if prob == 'simple1':
            x_star = np.array([1, 1])
            print('Optimal x is ', x_star)
            print('Distance from optimal point is ', np.linalg.norm(x_star - new_x))
        elif prob == 'simple2':
            print('First optimal point is (3, 2)')
            print('Second optimal point is (-2.8, 3.13)')
            print('Third optimal point is (-3.77, -3.28)')
            print('Fourth optimal point is (3.58, -1.84)')
        elif prob == 'simple3':
            print('The optimal point is (0, 0, 0, 0)')
        elif prob == 'test':
            x_star = np.array([0, 0])
            print('Optimal x is ', x_star)
            print('Distance from optimal point is ', np.linalg.norm(x_star - new_x))
        else:
            print('Optimal point not entered yet')
    return x_hist, new_x

