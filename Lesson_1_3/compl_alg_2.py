'''
Приклад аналізу складності алгоритму:
Алгоритм формування чисел ряду Фібоначчі

'''


def fibonacci(n : int) -> int:                          # Класичний алгоритм Фібоначчі - повертає число Фібоначчі = n
    """
    0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, ...
    Fn = (Fn-1) + (Fn-2)
    Класичний алгоритм:
    рекурсивне повторення операцій Фібоначчі для різних значень
    Time : O(2^n) - exponential / експоненційна нотація складності - наслідок рекурсії
    Space: O(n) - для рекурсивного ряду викликів
    """
    if n < 0:
        print("Incorrect input")
    elif n == 0:
        return 0
    elif n == 1 or n == 2:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
    
def fib_dynamic(n : int ) -> int:                       # Оптимізація алгоритму Фібоначчі - динамічне програмування
    """
    Dynamic programming example
    https://foxminded.ua/metod-dynamichnoho-prohramuvannia/
    створюється масив вхідних значень та який поповнюється динамічно.
    Time : O(n) - linear
    Space: O(n)
    """
    arr = [0, 1,]
    
    for i in range(2, n + 1):
        arr.insert(i, arr[i - 1] + arr[i - 2])
    
    return arr[-1] 


def fibonacci_space_optimized(n):
    """
    Алгоритм оптимізації використання ресурсів - пряме обчислення числа Фібоначчі c = f(f, b)
    Time : O(n)
    Space: O(1)
    """
    a = 0
    b = 1
    # Перевірте, чи n менше за 0
    if n < 0:
        print("Incorrect input")
         
    # Перевірте, чи дорівнює n 0
    elif n == 0:
        return 0
       
    # Перевірте, чи n дорівнює 1, інакше - число Фібоначчі = c = f(f, b)
    elif n == 1:
        return b
    else:
        for _ in range(1, n):
            c = a + b
            a = b
            b = c
        return b

if __name__ == "__main__":

    import cProfile         # імпорт модуля cProfile

    '''
    cProfile - модуль профайлер - інструмент Python: аналіз внутрішніх викликів інтерпретатора - оцінювання їх швидкодії.
    cProfileі profile - забезпечують детерміноване профілювання програм Python. 
    Профіль — це набір статистичних даних, який описує , як часто та як довго виконуються різні частини програми. 
    Цю статистику можна форматувати у звіти за допомогою pstats модуля.
    https://docs.python.org/3/library/profile.html
    https://www.toucantoco.com/en/tech-blog/python-performance-optimization
    '''



    number_to_test = 10                           # параметри для обрахунку

    # ---------------------   робота профайлера ----------------------------
    cp = cProfile.Profile()                         # створення профайлера
    cp.enable()                                     # увімкнути профайлер для аналізу функціонування сутностей, вказаних далі

    print('fibonacci(number_to_test) =', fibonacci(number_to_test))
    # print(fib_dynamic(number_to_test))
    # print(fibonacci_space_optimized(number_to_test))
    
    cp.disable()                                    # зупинка профайлера
    cp.print_stats()                                # відображення статистики профайлера

    # ---------------------   робота профайлера ----------------------------

    '''
    Результат профілювання:
    
    fibonacci(number_to_test) = 55                                                  - відгук функції - число Фібоначчі
         111 function calls (3 primitive calls) in 0.000 seconds                    - відстежено 111 викликів / дій, з низ 3 примітивних - не рекурсія, тощо.

   Ordered by: standard name                                                        - упорядковані дані за стандартним ім'ям

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    109/1    0.000    0.000    0.000    0.000 fibonacci.py:8(fibonacci)             - характеристика функції fibonacci
        1    0.000    0.000    0.000    0.000 {built-in method builtins.print}      - характеристика print
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}

    Значення індикаторів:    
    - ncalls - виклики - ПРОСТІР, результат 109/1 - функція рекурсувала: 109 - загальних викликів, 1 - первинних викликів;
    - tottime - загальноий ЧАС, витраченний на дану функцію / сутність (без підфункцій);
    - percall - є часткою tottime поділеного наncalls;
    - cumtime - загальний час, витрачений на цю та всі підфункції (від виклику до виходу);
    - percall - є часткою cumtime поділеного на примітивні виклики;
    - filename:lineno(function) - надає відповідні дані кожної функції / сутності.
        
    '''
