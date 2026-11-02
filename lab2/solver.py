"""Метод ломаных Пиявского. Пример запуска приведён в README.md."""
import argparse
import ast
import bisect
import csv
import json
import math
from pathlib import Path
from time import perf_counter


def parse_function(formula):
    """Разрешаем только арифметику, x и явно перечисленные функции."""
    names = {k: getattr(math, k) for k in ('sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'pi', 'e')}
    names['abs'] = abs
    tree = ast.parse(formula, mode='eval')
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name,
               ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub,
               ast.UAdd, ast.Call)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError('Недопустимая конструкция в формуле')
        if isinstance(node, ast.Name) and node.id not in names and node.id != 'x':
            raise ValueError('Неизвестное имя: ' + node.id)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or not callable(names.get(node.func.id)) or node.keywords:
                raise ValueError('Недопустимый вызов')
        if isinstance(node, ast.Constant) and (type(node.value) not in (int, float)):
            raise ValueError('Допустимы только числовые константы')
    code = compile(tree, '<formula>', 'eval')
    def function(x):
        value = float(eval(code, {'__builtins__': {}}, dict(names, x=x)))
        if not math.isfinite(value):
            raise ValueError('Функция должна иметь конечные значения')
        return value
    return function


def corners(points, L):
    lower = []
    for i, ((left, fl), (right, fr)) in enumerate(zip(points, points[1:])):
        if abs(fr - fl) > L * (right - left) + 1e-10 * max(1, abs(fl), abs(fr)):
            raise ValueError('L слишком мала: нарушено условие Липшица')
        x = (left + right) / 2 + (fl - fr) / (2 * L)
        p = (fl + fr - L * (right - left)) / 2
        lower.append((p, x, i))
    return lower


def piyavsky(function, a, b, eps, L, max_iterations=100000):
    if not all(math.isfinite(v) for v in (a, b, eps, L)) or not (a < b and eps > 0 and L > 0):
        raise ValueError('Нужно a < b, eps > 0, L > 0; все числа конечны')
    start = perf_counter()
    points = [(a, function(a)), (b, function(b))]
    history = []
    for iteration in range(1, max_iterations + 1):
        lower = corners(points, L)
        min_p = min(row[0] for row in lower)
        # При равенстве высот с учётом округления выбираем левую вершину.
        p, x, i = min((row for row in lower if row[0] <= min_p + 1e-12), key=lambda row: row[1])
        left, right = points[i][0], points[i + 1][0]
        if not left < x < right:
            raise RuntimeError('Новая точка совпала с границей: проверьте L и eps')
        fx = function(x)
        delta = fx - p
        history.append(dict(iteration=iteration, left=left, right=right, x=x, p=p, fx=fx, delta=delta))
        bisect.insort(points, (x, fx))
        # Проверяем также две новые пары до принятия результата.
        corners(points, L)
        if delta < eps:
            best_x, best_f = min(points, key=lambda t: t[1])
            return dict(x=best_x, fx=best_f, iterations=iteration,
                        evaluations=len(points), seconds=perf_counter()-start,
                        last_delta=delta, lower_bound=min_p,
                        certified_gap=best_f-min_p, points=points, history=history)
    raise RuntimeError('Достигнуто предельное число итераций без нужной точности')


def plot_result(function, a, b, L, result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Эта равномерная сетка используется ТОЛЬКО для рисования графика.
    xs = [a + (b-a)*i/2000 for i in range(2001)]
    points = result['points']
    broken_x, broken_y = [points[0][0]], [points[0][1]]
    for (p, x, _), (right, fr) in zip(corners(points, L), points[1:]):
        broken_x.extend([x, right]); broken_y.extend([p, fr])
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    for ax in axes:
        ax.plot(xs, [function(x) for x in xs], label='W(x)', color='#1f4d63')
        ax.plot(broken_x, broken_y, label='Итоговая ломаная', color='#b45309', linewidth=1)
        ax.scatter(*zip(*points), s=9, color='#555555', label='Пробы')
        ax.scatter([result['x']], [result['fx']], s=55, color='#a00025', zorder=5, label='Найденный минимум')
        ax.set_xlabel('x'); ax.set_ylabel('W(x)'); ax.grid(alpha=.25)
    axes[0].legend(fontsize=9)
    axes[0].set_title('Поиск глобального минимума: общий вид')
    axes[1].set_xlim(max(a, result['x']-.15), min(b, result['x']+.15))
    axes[1].set_ylim(result['fx']-.025, result['fx']+.3)
    axes[1].set_title('Увеличение около глобального минимума')
    fig.savefig(path, dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--function', default='x**2 + 10*(1-cos(2*pi*x))')
    parser.add_argument('--a', type=float, default=-2)
    parser.add_argument('--b', type=float, default=2)
    parser.add_argument('--eps', type=float, default=.01)
    parser.add_argument('--L', type=float, default=70, help='Обоснованная константа Липшица; для примера 70')
    parser.add_argument('--output', default='.')
    args = parser.parse_args()
    function = parse_function(args.function)
    result = piyavsky(function, args.a, args.b, args.eps, args.L)
    folder = Path(args.output); folder.mkdir(parents=True, exist_ok=True)
    with (folder/'iterations.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=result['history'][0].keys())
        writer.writeheader(); writer.writerows(result['history'])
    summary = {key: value for key, value in result.items() if key not in ('points', 'history')}
    summary.update(formula=args.function, a=args.a, b=args.b, eps=args.eps, L=args.L)
    (folder/'result.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    plot_result(function, args.a, args.b, args.L, result, folder/'solver-result.png')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
