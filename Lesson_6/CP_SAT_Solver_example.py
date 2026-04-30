#----------------------------- Decision Support System (DSS) -------------------------------------

'''

ДЕМОНСТРАЦІЙНИЙ ПРИКЛАД СКЛАДАННЯ РОЗКЛАДУ
-----------------------------------------------------------------------------------------------------------------------------------------

Проблема складання розкладів:
Однією з поширених проблем планування є робочий цех, у якому кілька завдань обробляються на кількох машинах.
Кожне завдання складається з послідовності часткових завдань, які повинні бути виконані в заданому порядку, і кожне завдання має бути оброблено на певній машині.
Наприклад, робота може полягати у виробництві одного предмета споживання, наприклад автомобіля.
Проблема полягає в плануванні завдань на машинах таким чином, щоб мінімізувати довжину розкладу — час, потрібний для виконання всіх завдань.

---------------------------------------------- Ознака / особливість класичних підходів -----------------------------------------------------
!!! ГОЛОВНА ЦІЛЬОВА ФУНКЦІЯ ЕФЕКТИВНОСТІ: мінімізація часу, потрібного для виконання всіх завдань.
!!! Решта вимог:    ОБМЕЖЕННЯ.
--------------------------------------------------------------------------------------------------------------------------------------------

Є кілька обмежень для проблеми складання розкладів:
Жодне завдання не можна розпочати, доки не буде виконано попереднє завдання для цього завдання.
Машина може працювати лише над одним завданням одночасно.
Завдання, яке було розпочате, має бути доведено до кінця.

Результат:
план - як підстава для побудови діаграми Ганта

Приклад:
# https://developers.google.com/optimization/scheduling/job_shop/#python_8

Інструкції із встановлення бібліотеки:
https://developers.google.com/optimizat ion/install/java/pkg_windows

'''

""" -------------- Minimal jobshop example ---------------------------"""


import collections
from ortools.sat.python import cp_model


def main():
    """Minimal jobshop problem."""
    # Data.
    jobs_data = [  # task = (machine_id, processing_time).
        [(0, 3), (1, 2), (2, 2)],  # Job0
        [(0, 2), (2, 1), (1, 4)],  # Job1
        [(1, 4), (2, 3)]  # Job2
    ]

    machines_count = 1 + max(task[0] for job in jobs_data for task in job)
    all_machines = range(machines_count)
    # Computes horizon dynamically as the sum of all durations.
    horizon = sum(task[1] for job in jobs_data for task in job)

    # ------------------------------------------------ Модель ---------------------------------------
    # Create the model.
    model = cp_model.CpModel()

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple('task_type', 'start end interval')
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index duration')

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            duration = task[1]
            suffix = '_%i_%i' % (job_id, task_id)
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                'interval' + suffix)
            all_tasks[job_id, task_id] = task_type(start=start_var,
                                                   end=end_var,
                                                   interval=interval_var)
            machine_to_intervals[machine].append(interval_var)

    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(all_tasks[job_id, task_id +
                                1].start >= all_tasks[job_id, task_id].end)

    #  # Precedences inside a job.
    # for job_id, job in enumerate(jobs_data):
    #      for task_id in range(len(job) - 1):
    #         model.Add(all_tasks[job_id, task_id +
    #                                     1].start == all_tasks[job_id, task_id].end)
    #         # model.Add(all_tasks[job_id, task_id +
    #         #                             1].start >= all_tasks[job_id, task_id].end)


    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job) - 1].end
        for job_id, job in enumerate(jobs_data)
    ])
    model.Minimize(obj_var)

    # ------------------------------------------------ Вирішувач ---------------------------------------
    # Creates the solver and solve.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print('Solution:')
        # Create one list of assigned tasks per machine.
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                assigned_jobs[machine].append(
                    assigned_task_type(start=solver.Value(
                        all_tasks[job_id, task_id].start),
                                       job=job_id,
                                       index=task_id,
                                       duration=task[1]))

        # Create per machine output lines.
        output = ''
        for machine in all_machines:
            # Sort by starting time.
            assigned_jobs[machine].sort()
            sol_line_tasks = 'Machine ' + str(machine) + ': '
            sol_line = '           '

            for assigned_task in assigned_jobs[machine]:
                name = 'job_%i_task_%i' % (assigned_task.job,
                                           assigned_task.index)
                # Add spaces to output to align columns.
                sol_line_tasks += '%-15s' % name

                start = assigned_task.start
                duration = assigned_task.duration
                sol_tmp = '[%i,%i]' % (start, start + duration)
                # Add spaces to output to align columns.
                sol_line += '%-15s' % sol_tmp

            sol_line += '\n'
            sol_line_tasks += '\n'
            output += sol_line_tasks
            output += sol_line

        # Finally print the solution found.
        print(f'Optimal Schedule Length: {solver.ObjectiveValue()}')
        print(output)
    else:
        print('No solution found.')

    # Statistics.
    print('\nStatistics')
    print('  - conflicts: %i' % solver.NumConflicts())
    print('  - branches : %i' % solver.NumBranches())
    print('  - wall time: %f s' % solver.WallTime())


if __name__ == '__main__':
    main()


'''
Рішення із документації:

 Optimal Schedule Length: 11
Machine 0: job_0_0   job_1_0
           [0,3]     [3,5]
Machine 1: job_2_0   job_0_1   job_1_2
           [0,4]     [4,6]     [7,11]
Machine 2: job_1_1   job_0_2   job_2_1
           [5,6]     [6,8]     [8,11]

'''