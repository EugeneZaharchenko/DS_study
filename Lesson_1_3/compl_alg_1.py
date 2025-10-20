
"""
Пошук коренів нелінійного рівняння методом метод ділення навпіл (дихотомії)  - bisection method:
Time Complexity: оцінити / знайти  самостійно
Auxiliary Space: оцінити / знайти  самостійно

"""

import cProfile
from scipy.optimize import fsolve
from compl_alg_model import graph_fun


def bisection(f, a, b, epsilon):

    # Перевірка наявності кореня на інтервалі
    if (f(a) * f(b) >= 0):
        print("Змініть межі пошуку коренів\n")
        return

    root = a

    # Контроль досяжності точності рішення
    while ((b - a) >= epsilon):

        # Ділення інтервалу навпіл
        root = (a + b) / 2

        # Перевірте, чи середня точка є коренем
        if (f(root) == 0.0):
            break

        # Вибір інтервалу пошуку коренів
        if (f(root) * f(a) < 0):
            b = root
        else:
            a = root

    return root


if __name__ == "__main__":

    # параметри інтегрування
    low = -10
    high = 0
    n = 100000
    epsilon = 1/n
    a = 2
    b = 1
    c = -6

    # Лямбда-функція — це анонімна функція, яка може приймати будь-яку кількість аргументів, але може мати лише один вираз
    # https://www.w3schools.com/python/python_lambda.asp
    fun = lambda x: a * x ** 2 + b * x + c
    graph_fun(fun, low, high, n)

    # Пошук коренів нелінійного рівняння
    print(bisection(fun, low, high, epsilon), ' ', epsilon)

    # Пошук коренів із бібліотекою scipy: Знайдіть корені функції.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html
    sol = fsolve(fun, [low, high])
    print(sol)

    # ---------------------- Аналітика складності алгоритму  ---------------------------
    cp = cProfile.Profile()         # використовуємо профайлер
    cp.enable()
    bisection(fun, low, high, epsilon)
    cp.disable()
    cp.print_stats()

