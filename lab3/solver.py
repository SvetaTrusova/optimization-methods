"""ЛР3. Метод Лагранжа. python solver.py [--output папка]"""
from pathlib import Path
import argparse
import json

parser = argparse.ArgumentParser(description='Метод Лагранжа, вариант 3')
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
output_dir = args.output.resolve()
output_dir.mkdir(parents=True, exist_ok=True)

import sympy as sp

x, y, lam = sp.symbols('x y lam', real=True)

variables = [x, y]

f = x + y

constraints = [x**2 + y**2 - 2]

L = f + lam * constraints[0]

equations = [sp.diff(L, v) for v in variables] + constraints

print('f =', f)

print('h =', constraints[0])

print('L =', L)

print('Equations (each equals zero):', equations)

J = sp.Matrix(constraints).jacobian(variables)

singular = sp.solve(constraints + list(J), variables, dict=True)

print('Singular feasible points:', singular)

solutions = sp.solve(equations, [x, y, lam], dict=True)

valid = []

for number, sol in enumerate(solutions, 1):

    if not all(sol[v].is_real is True for v in [x, y, lam]):

        continue

    residuals = [sp.simplify(h.subs(sol)) for h in constraints]

    stationarity = [sp.simplify(eq.subs(sol)) for eq in equations]

    feasible = all(r == 0 for r in residuals)

    value = sp.simplify(f.subs(sol))

    print('Candidate', number, sol)

    print('  constraint residuals =', residuals)

    print('  system residuals =', stationarity)

    print('  feasible =', feasible, '; f =', value)

    if feasible:

        valid.append((value, sol))

# For this variant: the circle is compact, f is continuous,

# all feasible points are regular, and the complete system was solved.

# Hence comparing all candidates gives the global extrema.

if valid:

    low = min(value for value, sol in valid)

    high = max(value for value, sol in valid)

    print('Global minimum:', low, [sol for value, sol in valid if value == low])

    print('Global maximum:', high, [sol for value, sol in valid if value == high])

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = output_dir
t = np.linspace(0, 2*np.pi, 1000)
x, y = np.sqrt(2)*np.cos(t), np.sqrt(2)*np.sin(t)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
ax = axes[0]
ax.plot(x, y, color='#246391', lw=2.2, label='Ограничение x² + y² = 2')
u = np.linspace(-1.8, 1.8, 300)
for c in [-1, 0, 1]:
    ax.plot(u, c-u, color='#a7adb4', lw=1, ls='--')
ax.plot(u, 2-u, color='#b54c35', lw=1.8, label='Уровень f = 2')
ax.plot(u, -2-u, color='#34835b', lw=1.8, label='Уровень f = −2')
for point, color, label, offset in [((1,1),'#b54c35','Максимум (1; 1)',(-110,14)),((-1,-1),'#34835b','Минимум (−1; −1)',(8,-22))]:
    ax.scatter(*point,color=color,s=55,zorder=5)
    ax.annotate(label,point,xytext=offset,textcoords='offset points',fontsize=9)
ax.set(xlim=(-1.8,1.8),ylim=(-1.8,1.8),xlabel='x',ylabel='y',title='Окружность и линии уровня f = x + y')
ax.set_aspect('equal'); ax.grid(alpha=.2); ax.legend(loc='lower left',fontsize=8)
ax = axes[1]
ax.plot(t,x+y,color='#246391',lw=2)
for angle, value, color in [(np.pi/4,2,'#b54c35'),(5*np.pi/4,-2,'#34835b')]:
    ax.scatter(angle,value,color=color,s=55,zorder=5)
    ax.annotate(f'f = {value}',(angle,value),xytext=(8,-18 if value>0 else 10),textcoords='offset points')
ax.axhline(2,color='#b54c35',ls='--',lw=1)
ax.axhline(-2,color='#34835b',ls='--',lw=1)
ax.set(xlim=(0,2*np.pi),ylim=(-2.5,2.5),xlabel='t, рад',ylabel='f(t)',title='Целевая функция вдоль окружности')
ax.set_xticks([0,np.pi/4,np.pi,5*np.pi/4,2*np.pi],['0','π/4','π','5π/4','2π'])
ax.grid(alpha=.2)
fig.savefig(out/'solver-result.png',dpi=220)
plt.close(fig)
print(out/'solver-result.png')


records = []
for value, sol in valid:
    records.append({
        'point': {str(v): str(sol[v]) for v in variables},
        'lambda': str(sol[lam]),
        'value': str(value),
        'constraint_residuals': [str(sp.simplify(h.subs(sol))) for h in constraints],
    })
result = {'variant': 3, 'f': str(f), 'constraint': str(constraints[0]),
          'candidates': records, 'minimum': str(low), 'maximum': str(high)}
(output_dir / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print('Results:', output_dir / 'result.json')
