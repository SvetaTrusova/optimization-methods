"""ЛР 1, вариант 7. Двухфазный симплекс-метод, x >= 0.

Запуск: python solver.py
Используется только стандартная библиотека Python.
"""

from fractions import Fraction


def reduced_costs(table, basis, costs):
    """Коэффициенты функции после исключения базисных переменных."""
    delta = [
        costs[j] - sum(costs[basis[i]] * row[j]
                       for i, row in enumerate(table))
        for j in range(len(costs))
    ]
    value = sum(costs[basis[i]] * row[-1]
                for i, row in enumerate(table))
    return delta, value


def pivot(table, basis, row_index, column):
    """Замена базиса методом Жордана-Гаусса."""
    element = table[row_index][column]
    table[row_index] = [v / element for v in table[row_index]]
    for i in range(len(table)):
        if i != row_index:
            factor = table[i][column]
            table[i] = [a - factor * b
                        for a, b in zip(table[i], table[row_index])]
    basis[row_index] = column


def print_table(state):
    print('\n' + state['title'])
    headings = ['Базис'] + state['names'] + ['b']
    rows = [[state['names'][state['basis'][i]]]
            + [str(v) for v in row] for i, row in enumerate(state['table'])]
    rows.append(['delta'] + [str(v) for v in state['delta']]
                + [str(state['value'])])
    widths = [max(len(r[j]) for r in [headings] + rows)
              for j in range(len(headings))]
    for row in [headings] + rows:
        print('  '.join(v.rjust(widths[j]) for j, v in enumerate(row)))


def simplex(table, basis, costs, names, phase, states, verbose):
    """Минимизация: вводим столбец с самым отрицательным delta."""
    for iteration in range(100):
        delta, value = reduced_costs(table, basis, costs)
        state = dict(title=f'{phase}, таблица {iteration}',
                     names=names[:], basis=basis[:],
                     table=[r[:] for r in table], delta=delta, value=value)
        states.append(state)
        if verbose:
            print_table(state)
        candidates = [j for j, d in enumerate(delta) if d < 0]
        if not candidates:
            return 'optimal', value
        column = min(candidates, key=lambda j: (delta[j], j))
        ratios = [(row[-1] / row[column], basis[i], i)
                  for i, row in enumerate(table) if row[column] > 0]
        if not ratios:
            return 'unbounded', None
        _, _, row_index = min(ratios)
        state['enter'] = names[column]
        state['leave'] = names[basis[row_index]]
        state['step'] = table[row_index][-1] / table[row_index][column]
        if verbose:
            print(f'Входит {state["enter"]}, выходит {state["leave"]}, '
                  f'шаг = {state["step"]}.')
        pivot(table, basis, row_index, column)
    raise RuntimeError('Превышено число итераций симплекс-метода')


def solve(verbose=True):
    # Порядок столбцов: x1, x2, x3, x4, s1, s2, a2, a3, b.
    names = ['x1', 'x2', 'x3', 'x4', 's1', 's2', 'a2', 'a3']
    rows = [[2, 1, 1, 0, 1, 0, 0, 0, 12],
            [0, 1, 1, 1, 0, 0, 1, 0, 9],
            [1, 0, 0, 1, 0, -1, 0, 1, 4]]
    table = [[Fraction(v) for v in row] for row in rows]
    basis = [4, 6, 7]
    states = []

    # Фаза I: минимизация суммы искусственных переменных.
    status, value = simplex(table, basis, [0, 0, 0, 0, 0, 0, 1, 1],
                            names, 'Фаза I: F = a2 + a3 -> min',
                            states, verbose)
    if status != 'optimal':
        raise RuntimeError('Не удалось завершить вспомогательную задачу')
    if value != 0:
        return dict(status='infeasible', states=states)

    # Убираем нулевые искусственные переменные из базиса при необходимости.
    for i in range(len(basis) - 1, -1, -1):
        if basis[i] >= 6:
            candidates = [j for j in range(6)
                          if j not in basis and table[i][j] != 0]
            if candidates:
                pivot(table, basis, i, candidates[0])
            else:
                del table[i]
                del basis[i]
    table = [row[:6] + [row[-1]] for row in table]
    names = names[:6]

    # Фаза II: max Z эквивалентна min W = -Z.
    status, value = simplex(table, basis, [-3, -2, -1, -4, 0, 0],
                            names, 'Фаза II: W = -Z -> min',
                            states, verbose)
    if status != 'optimal':
        return dict(status=status, states=states)
    solution = [Fraction(0)] * 6
    for i, column in enumerate(basis):
        solution[column] = table[i][-1]
    x = solution[:4]
    z = sum(c * v for c, v in zip([3, 2, 1, 4], x))
    # Проверка результата в исходных ограничениях.
    checks = [all(v >= 0 for v in x),
              2*x[0] + x[1] + x[2] <= 12,
              x[1] + x[2] + x[3] == 9,
              x[0] + x[3] >= 4, z == -value]
    if not all(checks):
        raise ArithmeticError('Полученная точка не прошла проверку')
    if verbose:
        print('\nОтвет: x* = (' + '; '.join(map(str, x)) + ')')
        print(f'Zmax = {z}')
        print('Все исходные ограничения выполнены.')
    return dict(status=status, x=x, z=z, states=states,
                slack=solution[4:])


if __name__ == '__main__':
    result = solve()
    if result['status'] == 'infeasible':
        print('Ограничения несовместны.')
    elif result['status'] == 'unbounded':
        print('Целевая функция не ограничена сверху.')
