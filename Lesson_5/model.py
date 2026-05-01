
'''
МОДЕЛЬ підінтегральної кривої.
'''

import numpy as np
import matplotlib.pyplot as plt


def irrational_f(x):
    return 1 / (1 + x ** 2)


def graph_irrational_fun(lo, hi, n):
    x = np.linspace(lo, hi, n)
    y = irrational_f(x)
    fig, ax = plt.subplots()
    plt.grid(True)
    ax.plot(x, y)
    plt.show()


def quadratic_f(a, b, c, x):
    return a * x ** 2 + b * x + c


def graph_quadratic_fun(a, b, c, lo, hi, n):
    x = np.linspace(lo, hi, n)
    y = quadratic_f(a, b, c, x)
    fig, ax = plt.subplots()
    plt.grid(True)
    ax.plot(x, y)
    plt.show()



def graph_fun(fun, lo, hi, n):
    x = np.linspace(lo, hi, n)
    y = fun(x)
    fig, ax = plt.subplots()
    plt.grid(True)
    ax.plot(x, y)
    plt.show()
