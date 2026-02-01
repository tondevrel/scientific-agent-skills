---
name: sympy
description: Comprehensive guide for SymPy - Python library for symbolic mathematics. Use for symbolic expressions, calculus (derivatives, integrals, limits, series), equation solving (algebraic, differential, systems), linear algebra, simplification, matrix operations, special functions, code generation, and mathematical proofs. Essential for analytical mathematics and computer algebra.
version: 1.13
license: BSD-3-Clause
---

# SymPy - Symbolic Mathematics

Python library for symbolic mathematics, providing computer algebra system (CAS) capabilities entirely in Python.

## When to Use

- Symbolic expressions and algebraic manipulation
- Calculus (derivatives, integrals, limits, series expansions)
- Solving equations (algebraic, transcendental, differential)
- Simplification and transformation of expressions
- Matrix operations and linear algebra (symbolic)
- Special mathematical functions
- Mathematical proofs and verification
- Code generation (C, Fortran, LaTeX)
- Physics calculations (mechanics, quantum mechanics)
- Number theory and discrete mathematics
- Logic and Boolean algebra
- Geometric algebra

## Reference Documentation

**Official docs**: https://docs.sympy.org/  
**Search patterns**: `sympy.symbols`, `sympy.diff`, `sympy.integrate`, `sympy.solve`, `sympy.simplify`

## Core Principles

### Use SymPy For

| Task | Module | Example |
|------|--------|---------|
| Symbols | `symbols` | `x, y = symbols('x y')` |
| Derivatives | `diff` | `diff(x**2, x)` |
| Integrals | `integrate` | `integrate(x**2, x)` |
| Equation solving | `solve` | `solve(x**2 - 4, x)` |
| Simplification | `simplify` | `simplify(expr)` |
| Limits | `limit` | `limit(sin(x)/x, x, 0)` |
| Series expansion | `series` | `series(exp(x), x, 0, 5)` |
| Matrices | `Matrix` | `Matrix([[1, 2], [3, 4]])` |

### Do NOT Use For

- Numerical computing (use NumPy, SciPy)
- Fast numerical calculations (symbolic is slow)
- Machine learning (use PyTorch, TensorFlow)
- Statistical analysis (use SciPy, statsmodels)
- Large-scale numerical simulations (use NumPy, Numba)

## Quick Reference

### Installation

```bash
# pip
pip install sympy

# conda
conda install sympy

# With all optional dependencies
pip install sympy[all]
```

### Standard Imports

```python
# Core imports
import sympy as sp
from sympy import symbols, Symbol, Function
from sympy import diff, integrate, limit, series

# Common operations
from sympy import simplify, expand, factor, collect
from sympy import solve, solveset, dsolve

# Special functions
from sympy import sin, cos, exp, log, sqrt, Abs
from sympy import pi, E, I, oo  # Constants

# Matrices
from sympy import Matrix, eye, zeros, ones

# Printing
from sympy import init_printing, pprint, latex
init_printing()  # Pretty printing in notebook
```

### Basic Pattern - Symbolic Expression

```python
from sympy import symbols, simplify, expand

# Define symbols
x, y, z = symbols('x y z')

# Create expression
expr = (x + y)**2

# Expand
expanded = expand(expr)
print(f"Expanded: {expanded}")  # x**2 + 2*x*y + y**2

# Factor back
from sympy import factor
factored = factor(expanded)
print(f"Factored: {factored}")  # (x + y)**2

# Substitute values
result = expr.subs([(x, 2), (y, 3)])
print(f"Result: {result}")  # 25
```

### Basic Pattern - Calculus

```python
from sympy import symbols, diff, integrate, limit
from sympy import sin, cos, exp

x = symbols('x')

# Derivative
f = x**3 + 2*x**2 - x + 1
df = diff(f, x)
print(f"f'(x) = {df}")  # 3*x**2 + 4*x - 1

# Integral
integral = integrate(sin(x), x)
print(f"∫sin(x)dx = {integral}")  # -cos(x)

# Definite integral
definite = integrate(x**2, (x, 0, 1))
print(f"∫₀¹ x²dx = {definite}")  # 1/3

# Limit
lim = limit(sin(x)/x, x, 0)
print(f"lim(sin(x)/x) as x→0 = {lim}")  # 1
```

## Critical Rules

### ✅ DO

- **Use symbols() for variables** - Define symbolic variables properly
- **Simplify expressions** - Use simplify() to clean up results
- **Use rational numbers** - Use Rational(1, 2) instead of 0.5
- **Check assumptions** - Set assumptions on symbols when needed
- **Use subs() for substitution** - Replace symbols with values
- **Pretty print results** - Use pprint() or init_printing()
- **Verify results numerically** - Convert to float for checking
- **Use appropriate functions** - Choose right solving function
- **Factor before solving** - Simplify equations first
- **Use lambdify for speed** - Convert to NumPy functions

### ❌ DON'T

- **Mix symbolic and numeric carelessly** - Be explicit with types
- **Use Python floats in symbolic** - Use Rational or Integer
- **Forget to define symbols** - Must declare before use
- **Ignore symbolic/numeric distinction** - Know when to use each
- **Use == for equation solving** - Use solve() or Eq()
- **Evaluate expensive operations blindly** - Some integrals are hard
- **Assume automatic simplification** - Often need explicit simplify()
- **Use symbolic for large numerical tasks** - Too slow
- **Forget assumptions** - Can affect results (positive, real, etc.)
- **Over-rely on solve()** - Use solveset() for better handling

## Anti-Patterns (NEVER)

```python
from sympy import symbols, solve, simplify, integrate, Rational
import sympy as sp

# ❌ BAD: Using float instead of Rational
x = symbols('x')
expr = x + 0.5  # Float!
# Result may not be exact

# ✅ GOOD: Use Rational for exact arithmetic
expr = x + Rational(1, 2)
# Exact representation

# ❌ BAD: Not defining symbols
result = y**2 + 2*y + 1  # NameError: y not defined

# ✅ GOOD: Define symbols first
y = symbols('y')
result = y**2 + 2*y + 1

# ❌ BAD: Using Python == for equations
solve(x**2 == 4)  # Wrong! Returns boolean

# ✅ GOOD: Use solve() properly
solve(x**2 - 4, x)  # Returns [-2, 2]

# Or use Eq()
from sympy import Eq
solve(Eq(x**2, 4), x)

# ❌ BAD: Not simplifying
expr = (x + 1)**2 - (x**2 + 2*x + 1)
print(expr)  # Messy, not simplified to 0

# ✅ GOOD: Simplify
result = simplify(expr)
print(result)  # 0

# ❌ BAD: Using symbolic for numerical loops
for i in range(1000000):
    result = sp.sin(sp.pi * i / 180)  # Very slow!

# ✅ GOOD: Lambdify for numerical work
import numpy as np
x = symbols('x')
f_sym = sp.sin(x)
f_num = sp.lambdify(x, f_sym, 'numpy')
results = f_num(np.linspace(0, np.pi, 1000000))  # Fast!

# ❌ BAD: Ignoring assumptions
x = symbols('x')
sqrt(x**2)  # Returns sqrt(x**2), not x (could be negative)

# ✅ GOOD: Set assumptions
x = symbols('x', positive=True)
simplify(sqrt(x**2))  # Returns x
```

## Symbols and Expressions

### Creating Symbols

```python
from sympy import symbols, Symbol

# Single symbol
x = Symbol('x')

# Multiple symbols
x, y, z = symbols('x y z')

# With assumptions
x = symbols('x', real=True)
y = symbols('y', positive=True)
n = symbols('n', integer=True)
theta = symbols('theta', real=True)

# Complex symbols
z = symbols('z', complex=True)

# Functions
from sympy import Function
f = Function('f')
g = Function('g')

# Indexed symbols
from sympy import IndexedBase, Idx
A = IndexedBase('A')
i, j = symbols('i j', integer=True)
element = A[i, j]

print(f"Assumptions for x: {x.assumptions0}")
```

### Building Expressions

```python
from sympy import symbols, sin, cos, exp, log, sqrt
from sympy import pi, E, I, oo

x, y, z = symbols('x y z')

# Arithmetic
expr1 = x + 2*y - 3*z
expr2 = x**2 + y**2
expr3 = x*y / z

# Functions
expr4 = sin(x) + cos(y)
expr5 = exp(x**2)
expr6 = log(x + 1)

# Constants
expr7 = pi * x
expr8 = E**x
expr9 = I * x  # Imaginary unit

# Complex expressions
expr10 = (x + y)**3 / (sqrt(x**2 + y**2))
expr11 = sin(x)**2 + cos(x)**2

# Accessing parts of expressions
print(f"Numerator: {expr3.as_numer_denom()[0]}")
print(f"Denominator: {expr3.as_numer_denom()[1]}")

# Expression info
print(f"Free symbols: {expr10.free_symbols}")
print(f"Is polynomial: {expr1.is_polynomial()}")
```

### Substitution

```python
from sympy import symbols, sin, cos, pi

x, y = symbols('x y')

# Basic substitution
expr = x**2 + 2*x + 1
result = expr.subs(x, 3)
print(f"Result: {result}")  # 16

# Multiple substitutions
expr = x**2 + y**2
result = expr.subs([(x, 1), (y, 2)])
print(f"Result: {result}")  # 5

# Substitute expression
expr = sin(x) + cos(x)
result = expr.subs(x, pi/4)
print(f"Result: {result}")  # sqrt(2)

# Sequential substitution
expr = x + y
temp = expr.subs(x, y)
result = temp.subs(y, 1)
print(f"Result: {result}")  # 2

# Substitute and simplify
from sympy import simplify
expr = sin(x)**2 + cos(x)**2
result = simplify(expr.subs(x, pi/3))
print(f"Result: {result}")  # 1
```

## Calculus

### Derivatives

```python
from sympy import symbols, diff, sin, cos, exp, log
from sympy import pprint

x, y, z = symbols('x y z')

# First derivative
f = x**3 + 2*x**2 - x + 1
df = diff(f, x)
print(f"f'(x) = {df}")  # 3*x**2 + 4*x - 1

# Higher order derivatives
d2f = diff(f, x, 2)  # Second derivative
d3f = diff(f, x, 3)  # Third derivative
print(f"f''(x) = {d2f}")
print(f"f'''(x) = {d3f}")

# Partial derivatives
f = x**2*y + y**3*z
df_dx = diff(f, x)
df_dy = diff(f, y)
df_dz = diff(f, z)
print(f"∂f/∂x = {df_dx}")  # 2*x*y
print(f"∂f/∂y = {df_dy}")  # x**2 + 3*y**2*z

# Mixed partial derivatives
d2f_dxdy = diff(f, x, y)
d2f_dydz = diff(f, y, z)
print(f"∂²f/∂x∂y = {d2f_dxdy}")  # 2*x

# Derivative of special functions
g = sin(x**2) * exp(-x)
dg = diff(g, x)
print(f"g'(x) = {dg}")

# Chain rule automatically applied
h = sin(cos(x))
dh = diff(h, x)
print(f"h'(x) = {dh}")  # -sin(x)*cos(cos(x))
```

### Integrals

```python
from sympy import symbols, integrate, sin, cos, exp, log, sqrt
from sympy import pi, oo

x, y = symbols('x y')

# Indefinite integrals
f = x**2
F = integrate(f, x)
print(f"∫x²dx = {F}")  # x**3/3

# Multiple integrals
f = x*y
F = integrate(f, x)
print(f"∫xy dx = {F}")  # x**2*y/2

# Definite integrals
result = integrate(x**2, (x, 0, 1))
print(f"∫₀¹ x²dx = {result}")  # 1/3

# Improper integrals
result = integrate(exp(-x), (x, 0, oo))
print(f"∫₀^∞ e^(-x)dx = {result}")  # 1

# Trigonometric integrals
result = integrate(sin(x)**2, x)
print(f"∫sin²(x)dx = {result}")  # x/2 - sin(2*x)/4

# Integration by parts (automatic)
result = integrate(x*exp(x), x)
print(f"∫x·e^x dx = {result}")

# Multiple integration
f = x*y
result = integrate(f, (x, 0, 1), (y, 0, 2))
print(f"∫∫xy dx dy = {result}")  # 1

# Difficult integrals
from sympy import erf
result = integrate(exp(-x**2), (x, -oo, oo))
print(f"∫₋∞^∞ e^(-x²)dx = {result}")  # sqrt(pi)
```

### Limits

```python
from sympy import symbols, limit, sin, cos, exp, log
from sympy import oo, pi

x = symbols('x')

# Basic limits
lim = limit(sin(x)/x, x, 0)
print(f"lim(sin(x)/x) as x→0 = {lim}")  # 1

# Limits at infinity
lim = limit((1 + 1/x)**x, x, oo)
print(f"lim((1+1/x)^x) as x→∞ = {lim}")  # E

# One-sided limits
lim_left = limit(1/x, x, 0, '-')
lim_right = limit(1/x, x, 0, '+')
print(f"Left limit: {lim_left}")   # -oo
print(f"Right limit: {lim_right}")  # oo

# Limits with L'Hôpital's rule (automatic)
lim = limit((exp(x) - 1)/x, x, 0)
print(f"lim((e^x - 1)/x) as x→0 = {lim}")  # 1

# Multivariable limits
y = symbols('y')
lim = limit(limit(x*y/(x**2 + y**2), x, 0), y, 0)
print(f"Double limit: {lim}")

# Limit of sequences
n = symbols('n', integer=True, positive=True)
lim = limit((1 + 1/n)**n, n, oo)
print(f"lim((1+1/n)^n) as n→∞ = {lim}")  # E
```

### Series Expansions

```python
from sympy import symbols, series, sin, cos, exp, log
from sympy import O  # Big-O notation

x = symbols('x')

# Taylor series
s = series(exp(x), x, 0, 5)
print(f"exp(x) ≈ {s}")
# 1 + x + x**2/2 + x**3/6 + x**4/24 + O(x**5)

# Remove O term
s_no_o = s.removeO()
print(f"Without O term: {s_no_o}")

# Series around different points
s = series(log(x), x, 1, 5)
print(f"log(x) around x=1: {s}")

# Trigonometric series
s = series(sin(x), x, 0, 7)
print(f"sin(x) ≈ {s}")
# x - x**3/6 + x**5/120 + O(x**7)

# Laurent series (negative powers)
s = series(1/(x**2 + x), x, 0, 3)
print(f"1/(x²+x) ≈ {s}")

# Series substitution
s1 = series(exp(x), x, 0, 5)
s2 = series(sin(x), x, 0, 5)
s_composed = series(s1.removeO().subs(x, s2.removeO()), x, 0, 5)
print(f"exp(sin(x)) ≈ {s_composed}")

# Multivariable series
y = symbols('y')
s = series(exp(x*y), x, 0, 3)
print(f"exp(xy) in x: {s}")
```

## Equation Solving

### Algebraic Equations

```python
from sympy import symbols, solve, Eq
from sympy import sqrt

x, y, z = symbols('x y z')

# Simple equation
solutions = solve(x**2 - 4, x)
print(f"x² = 4: {solutions}")  # [-2, 2]

# Using Eq() for clarity
eq = Eq(x**2, 4)
solutions = solve(eq, x)
print(f"Solutions: {solutions}")

# Multiple solutions
eq = x**3 - 6*x**2 + 11*x - 6
solutions = solve(eq, x)
print(f"x³ - 6x² + 11x - 6 = 0: {solutions}")  # [1, 2, 3]

# Quadratic formula (symbolic)
a, b, c = symbols('a b c')
eq = a*x**2 + b*x + c
solutions = solve(eq, x)
print(f"ax² + bx + c = 0:")
for sol in solutions:
    print(f"  x = {sol}")

# Rational equations
eq = 1/x + 1/(x+1) - 1/2
solutions = solve(eq, x)
print(f"Solutions: {solutions}")

# Equations with radicals
eq = sqrt(x) + sqrt(x - 1) - 2
solutions = solve(eq, x)
print(f"Solutions: {solutions}")
```

### Systems of Equations

```python
from sympy import symbols, solve, Eq

x, y, z = symbols('x y z')

# Linear system
eq1 = Eq(x + y, 5)
eq2 = Eq(x - y, 1)
solution = solve([eq1, eq2], [x, y])
print(f"Linear system: {solution}")  # {x: 3, y: 2}

# Nonlinear system
eq1 = x**2 + y**2 - 4
eq2 = x - y - 1
solutions = solve([eq1, eq2], [x, y])
print(f"Nonlinear system: {solutions}")

# Three equations, three unknowns
eq1 = x + y + z - 6
eq2 = 2*x - y + z - 2
eq3 = x + 2*y - z - 2
solution = solve([eq1, eq2, eq3], [x, y, z])
print(f"3x3 system: {solution}")

# Parametric solutions
eq1 = x + 2*y + z - 1
eq2 = 2*x + y + 2*z - 2
# Underdetermined system
solution = solve([eq1, eq2], [x, y, z])
print(f"Parametric solution: {solution}")
```

### Differential Equations

```python
from sympy import symbols, Function, dsolve, Eq
from sympy import diff, sin, cos, exp

x = symbols('x')
f = Function('f')

# First-order ODE
# f'(x) = f(x)
eq = Eq(diff(f(x), x), f(x))
solution = dsolve(eq, f(x))
print(f"f'(x) = f(x): {solution}")  # f(x) = C1*exp(x)

# Second-order ODE
# f''(x) + f(x) = 0
eq = Eq(diff(f(x), x, 2) + f(x), 0)
solution = dsolve(eq, f(x))
print(f"f''(x) + f(x) = 0: {solution}")
# f(x) = C1*sin(x) + C2*cos(x)

# ODE with initial conditions
eq = Eq(diff(f(x), x), f(x))
ics = {f(0): 1}  # Initial condition: f(0) = 1
solution = dsolve(eq, f(x), ics=ics)
print(f"Solution with IC: {solution}")  # f(x) = exp(x)

# Non-homogeneous ODE
# f''(x) + 4*f(x) = sin(x)
eq = Eq(diff(f(x), x, 2) + 4*f(x), sin(x))
solution = dsolve(eq, f(x))
print(f"Non-homogeneous: {solution}")

# System of ODEs
g = Function('g')
eq1 = Eq(diff(f(x), x), g(x))
eq2 = Eq(diff(g(x), x), -f(x))
solution = dsolve([eq1, eq2], [f(x), g(x)])
print(f"System of ODEs: {solution}")
```

### Transcendental and Numerical

```python
from sympy import symbols, solve, sin, cos, exp, log
from sympy import nsolve, lambdify
import numpy as np

x = symbols('x')

# Transcendental equation
eq = sin(x) - x/2
solutions = solve(eq, x)
print(f"sin(x) = x/2 solutions: {solutions}")

# Some equations can't be solved symbolically
# Use nsolve for numerical solution
eq = exp(x) + x
# nsolve requires initial guess
try:
    solution = nsolve(eq, -1)  # Start near -1
    print(f"e^x + x = 0: x ≈ {solution}")
except:
    print("Could not find solution")

# Trigonometric equation
eq = sin(x) + cos(x) - 1
solutions = solve(eq, x)
print(f"sin(x) + cos(x) = 1: {solutions[:3]}")  # First 3

# Logarithmic equation
eq = log(x) - 1/x
solutions = solve(eq, x)
print(f"log(x) = 1/x: {solutions}")
```

## Simplification and Manipulation

### Simplification

```python
from sympy import symbols, simplify, sin, cos, exp, log, sqrt
from sympy import pi

x, y = symbols('x y')

# Basic simplification
expr = (x**2 + 2*x + 1) / (x + 1)
simplified = simplify(expr)
print(f"Simplified: {simplified}")  # x + 1

# Trigonometric simplification
expr = sin(x)**2 + cos(x)**2
simplified = simplify(expr)
print(f"sin²x + cos²x = {simplified}")  # 1

# Exponential simplification
expr = exp(x) * exp(y)
simplified = simplify(expr)
print(f"e^x · e^y = {simplified}")  # exp(x + y)

# Radical simplification
expr = sqrt(x**2)
x_pos = symbols('x', positive=True)
simplified = simplify(expr.subs(x, x_pos))
print(f"√(x²) = {simplified}")  # x (if x positive)

# Complex simplification
expr = (x + y)**3 - (x**3 + 3*x**2*y + 3*x*y**2 + y**3)
simplified = simplify(expr)
print(f"Result: {simplified}")  # 0

# Logarithmic simplification
expr = log(x) + log(y)
simplified = simplify(expr)
print(f"log(x) + log(y) = {simplified}")  # log(x*y)
```

### Expansion and Factoring

```python
from sympy import symbols, expand, factor, collect
from sympy import sin, cos

x, y, z = symbols('x y z')

# Expand polynomial
expr = (x + y)**3
expanded = expand(expr)
print(f"(x+y)³ = {expanded}")
# x**3 + 3*x**2*y + 3*x*y**2 + y**3

# Expand trigonometric
expr = sin(x + y)
expanded = expand(expr, trig=True)
print(f"sin(x+y) = {expanded}")
# sin(x)*cos(y) + sin(y)*cos(x)

# Factor polynomial
expr = x**2 + 2*x + 1
factored = factor(expr)
print(f"x² + 2x + 1 = {factored}")  # (x + 1)**2

# Factor complex expressions
expr = x**4 - 1
factored = factor(expr)
print(f"x⁴ - 1 = {factored}")
# (x - 1)*(x + 1)*(x**2 + 1)

# Collect terms
expr = x*y + x - 3 + 2*x**2 - z*x**2 + x**3
collected = collect(expr, x)
print(f"Collected: {collected}")
# x**3 + x**2*(2 - z) + x*(y + 1) - 3

# Expand and factor
expr = (x + 1)*(x + 2)*(x + 3)
expanded = expand(expr)
refactored = factor(expanded)
print(f"Expanded: {expanded}")
print(f"Factored: {refactored}")
```

### Rewriting Expressions

```python
from sympy import symbols, sin, cos, exp, log, tan
from sympy import sinh, cosh

x = symbols('x')

# Rewrite in terms of different functions
expr = tan(x)
rewritten = expr.rewrite(sin)
print(f"tan(x) in terms of sin: {rewritten}")
# sin(x)/cos(x)

# Exponential form
expr = sin(x)
rewritten = expr.rewrite(exp)
print(f"sin(x) in exp form: {rewritten}")
# -I*(exp(I*x) - exp(-I*x))/2

# Hyperbolic to exponential
expr = sinh(x)
rewritten = expr.rewrite(exp)
print(f"sinh(x) = {rewritten}")
# (exp(x) - exp(-x))/2

# Logarithm base change
expr = log(x, 10)  # log base 10
rewritten = expr.rewrite(log)  # Natural log
print(f"log₁₀(x) = {rewritten}")

# Trigonometric identities
expr = sin(2*x)
rewritten = expr.rewrite(sin, cos)
print(f"sin(2x) = {rewritten}")  # 2*sin(x)*cos(x)
```

### Partial Fractions

```python
from sympy import symbols, apart, together
from sympy import cancel, factor

x = symbols('x')

# Partial fraction decomposition
expr = (x**2 + 2*x + 1) / (x**3 + x**2)
partial = apart(expr, x)
print(f"Partial fractions: {partial}")
# 1/x + 2/x**2 - 1/(x + 1)

# Recombine fractions
expr = 1/x + 2/x**2 - 1/(x + 1)
combined = together(expr)
print(f"Combined: {combined}")

# Cancel common factors
expr = (x**2 - 1) / (x - 1)
cancelled = cancel(expr)
print(f"Cancelled: {cancelled}")  # x + 1

# More complex example
expr = (4*x**3 + 21*x**2 + 10*x + 12) / (x**4 + 5*x**3 + 5*x**2 + 4*x)
partial = apart(expr, x)
print(f"Complex partial fractions: {partial}")
```

## Linear Algebra

### Matrices

```python
from sympy import Matrix, eye, zeros, ones, diag
from sympy import symbols

x, y, z = symbols('x y z')

# Create matrices
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[x, y], [z, x+y]])

print(f"Matrix A:\n{A}")

# Special matrices
I = eye(3)  # Identity
Z = zeros(2, 3)  # Zeros
O = ones(3, 2)  # Ones
D = diag(1, 2, 3)  # Diagonal

print(f"Identity:\n{I}")
print(f"Diagonal:\n{D}")

# Matrix operations
C = A + A
D = A * A  # Matrix multiplication
E = 2 * A

print(f"A + A:\n{C}")
print(f"A * A:\n{D}")

# Transpose
At = A.T
print(f"A transpose:\n{At}")

# Symbolic matrices
M = Matrix([[x, y], [y, z]])
print(f"Symbolic matrix:\n{M}")
```

### Matrix Properties

```python
from sympy import Matrix, symbols
from sympy import sqrt

# Create matrix
A = Matrix([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]])

# Determinant
det_A = A.det()
print(f"det(A) = {det_A}")  # 0 (singular)

# Rank
rank_A = A.rank()
print(f"rank(A) = {rank_A}")  # 2

# Trace
trace_A = A.trace()
print(f"trace(A) = {trace_A}")  # 15

# Inverse (if exists)
B = Matrix([[1, 2], [3, 4]])
B_inv = B.inv()
print(f"B inverse:\n{B_inv}")

# Verify: B * B^(-1) = I
product = B * B_inv
print(f"B * B⁻¹:\n{product}")

# Eigenvalues and eigenvectors
eigenvals = B.eigenvals()
print(f"Eigenvalues: {eigenvals}")

eigenvects = B.eigenvects()
print(f"Eigenvectors:")
for val, mult, vects in eigenvects:
    print(f"  λ = {val}, multiplicity = {mult}")
    for v in vects:
        print(f"    v = {v}")
```

### Matrix Decompositions

```python
from sympy import Matrix, eye
from sympy import QR, LU

# QR decomposition
A = Matrix([[1, 2], [3, 4], [5, 6]])
Q, R = A.QRdecomposition()
print(f"Q:\n{Q}")
print(f"R:\n{R}")

# Verify: A = Q*R
print(f"Q*R:\n{Q*R}")

# LU decomposition
B = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
L, U, perm = B.LUdecomposition()
print(f"L:\n{L}")
print(f"U:\n{U}")

# Jordan form
from sympy import Matrix
M = Matrix([[2, 1], [0, 2]])
P, J = M.jordan_form()
print(f"Jordan form J:\n{J}")
print(f"Transform P:\n{P}")

# Diagonalization (if possible)
A = Matrix([[1, 2], [2, 1]])
try:
    P, D = A.diagonalize()
    print(f"Diagonal form D:\n{D}")
    print(f"Transform P:\n{P}")
    # Verify: A = P*D*P^(-1)
    print(f"P*D*P⁻¹:\n{P*D*P.inv()}")
except:
    print("Matrix not diagonalizable")
```

### Solving Linear Systems

```python
from sympy import Matrix, symbols

x, y, z = symbols('x y z')

# System: Ax = b
A = Matrix([[3, 2, -1],
            [2, -2, 4],
            [-1, 0.5, -1]])

b = Matrix([1, -2, 0])

# Solve using matrix inverse
x_sol = A.inv() * b
print(f"Solution using inverse:\n{x_sol}")

# Solve using solve_linear_system
from sympy import solve_linear_system
# Augmented matrix
M = Matrix([[3, 2, -1, 1],
            [2, -2, 4, -2],
            [-1, 0.5, -1, 0]])

solution = solve_linear_system(M, x, y, z)
print(f"Solution: {solution}")

# Row reduction (RREF)
A_extended = Matrix([[3, 2, -1, 1],
                     [2, -2, 4, -2],
                     [-1, 0.5, -1, 0]])

rref_form, pivot_cols = A_extended.rref()
print(f"RREF:\n{rref_form}")
print(f"Pivot columns: {pivot_cols}")
```

## Special Functions and Constants

### Mathematical Constants

```python
from sympy import pi, E, I, oo, zoo
from sympy import GoldenRatio, EulerGamma
from sympy import S  # Singleton for exact numbers

# Pi
print(f"π = {pi}")
print(f"π ≈ {pi.evalf()}")  # Numerical value

# Euler's number
print(f"e = {E}")
print(f"e ≈ {E.evalf()}")

# Imaginary unit
print(f"i = {I}")
print(f"i² = {I**2}")  # -1

# Infinity
print(f"∞ = {oo}")
print(f"Complex infinity = {zoo}")

# Golden ratio
print(f"φ = {GoldenRatio}")
print(f"φ ≈ {GoldenRatio.evalf()}")

# Euler-Mascheroni constant
print(f"γ = {EulerGamma}")
print(f"γ ≈ {EulerGamma.evalf()}")

# Exact rational numbers
from sympy import Rational
half = Rational(1, 2)
third = Rational(1, 3)
print(f"1/2 + 1/3 = {half + third}")  # 5/6

# Infinity arithmetic
print(f"1/∞ = {1/oo}")  # 0
print(f"∞ + 1 = {oo + 1}")  # oo
```

### Special Functions

```python
from sympy import symbols, factorial, binomial, fibonacci
from sympy import gamma, beta, zeta
from sympy import besselj, bessely

n, m, x = symbols('n m x')

# Factorial
print(f"5! = {factorial(5)}")  # 120
print(f"n! = {factorial(n)}")

# Binomial coefficient
print(f"C(5,2) = {binomial(5, 2)}")  # 10
print(f"C(n,m) = {binomial(n, m)}")

# Fibonacci
print(f"F(10) = {fibonacci(10)}")  # 55

# Gamma function
print(f"Γ(5) = {gamma(5)}")  # 24 = 4!
print(f"Γ(1/2) = {gamma(S(1)/2)}")  # sqrt(pi)

# Beta function
print(f"B(2,3) = {beta(2, 3)}")

# Riemann zeta function
print(f"ζ(2) = {zeta(2)}")  # pi**2/6
print(f"ζ(2) ≈ {zeta(2).evalf()}")

# Bessel functions
print(f"J₀(x) = {besselj(0, x)}")
print(f"J₁(x) = {besselj(1, x)}")
```

### Error and Hypergeometric Functions

```python
from sympy import symbols, erf, erfc, erfi
from sympy import hyper, meijerg
from sympy import exp, sqrt, pi

x = symbols('x')

# Error function
print(f"erf(x) = {erf(x)}")
print(f"erf(1) ≈ {erf(1).evalf()}")

# Complementary error function
print(f"erfc(x) = {erfc(x)}")
print(f"erfc(x) = {1 - erf(x)}")  # Relationship

# Imaginary error function
print(f"erfi(x) = {erfi(x)}")

# Integral representation
from sympy import integrate
erf_integral = 2/sqrt(pi) * integrate(exp(-x**2), (x, 0, x))
print(f"erf via integral: {erf_integral}")

# Hypergeometric function
# 2F1(a, b; c; z)
result = hyper([1, 2], [3], x)
print(f"Hypergeometric: {result}")
```

## Code Generation

### Lambdify (Convert to NumPy)

```python
from sympy import symbols, sin, cos, exp, lambdify
import numpy as np
import matplotlib.pyplot as plt

x = symbols('x')

# Symbolic expression
expr = sin(x) * exp(-x**2/2)

# Convert to numerical function
f = lambdify(x, expr, 'numpy')

# Evaluate on array
x_vals = np.linspace(-3, 3, 1000)
y_vals = f(x_vals)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(r'$f(x) = \sin(x) \cdot e^{-x^2/2}$')
plt.grid(True)
plt.savefig('lambdify_example.png', dpi=150)

print(f"Function evaluated at x=1: {f(1):.6f}")

# Multiple arguments
x, y = symbols('x y')
expr = x**2 + y**2
f = lambdify((x, y), expr, 'numpy')

X, Y = np.meshgrid(np.linspace(-2, 2, 50),
                    np.linspace(-2, 2, 50))
Z = f(X, Y)

print(f"Shape of result: {Z.shape}")
```

### LaTeX Generation

```python
from sympy import symbols, sin, cos, exp, integrate, Integral
from sympy import latex, Derivative, limit

x, y = symbols('x y')

# Convert expression to LaTeX
expr = sin(x)**2 + cos(x)**2
latex_str = latex(expr)
print(f"LaTeX: {latex_str}")

# Integral in LaTeX
expr = integrate(x**2, x)
latex_str = latex(expr)
print(f"∫x²dx in LaTeX: {latex_str}")

# Unevaluated integral
expr = Integral(sin(x) * exp(-x), (x, 0, float('inf')))
latex_str = latex(expr)
print(f"Integral notation: {latex_str}")

# Derivative in LaTeX
expr = Derivative(sin(x**2), x)
latex_str = latex(expr)
print(f"Derivative: {latex_str}")

# Complex expression
expr = limit((1 + 1/x)**x, x, float('inf'))
latex_str = latex(expr)
print(f"Limit: {latex_str}")

# Matrix in LaTeX
from sympy import Matrix
M = Matrix([[x, y], [y, x]])
latex_str = latex(M)
print(f"Matrix:\n{latex_str}")
```

### Code Generation (C, Fortran)

```python
from sympy import symbols, sin, cos, exp
from sympy.utilities.codegen import codegen

x, y = symbols('x y')

# Expression to convert
expr = sin(x) * exp(-y)

# Generate C code
[(c_name, c_code), (h_name, c_header)] = codegen(
    ('func', expr), 'C', 'file', header=False
)

print("C code:")
print(c_code)

# Generate Fortran code
[(f_name, f_code)] = codegen(
    ('func', expr), 'F95', 'file', header=False
)

print("\nFortran code:")
print(f_code)

# More complex: function with multiple outputs
from sympy import symbols, Function, Derivative
from sympy.utilities.autowrap import autowrap

x = symbols('x')
expr1 = x**2
expr2 = x**3

# Can use autowrap for automatic compilation
# (requires compiler)
# f = autowrap(expr1, language='C', backend='Cython')
```

### SymPy to NumPy Functions

```python
from sympy import symbols, Matrix, lambdify
import numpy as np

x, y = symbols('x y')

# Matrix operations
M = Matrix([[x, y], [y, x]])

# Convert matrix to function
M_func = lambdify((x, y), M, 'numpy')

# Evaluate
result = M_func(1, 2)
print(f"Matrix at (1, 2):\n{result}")

# Matrix with operations
expr = M.det()  # Determinant
det_func = lambdify((x, y), expr, 'numpy')

print(f"Determinant at (1, 2): {det_func(1, 2)}")

# Vector operations
v = Matrix([x**2, y**2, x*y])
v_func = lambdify((x, y), v, 'numpy')

result = v_func(np.array([1, 2, 3]), np.array([4, 5, 6]))
print(f"Vector result:\n{result}")
```

## Physics and Engineering

### Classical Mechanics

```python
from sympy import symbols, Function, diff, solve, Eq
from sympy import sin, cos, sqrt, simplify

t = symbols('t', positive=True)
m, g, L = symbols('m g L', positive=True)

# Simple pendulum
theta = Function('theta')

# Equation of motion: theta'' + (g/L)*sin(theta) = 0
eq = Eq(diff(theta(t), t, 2) + (g/L)*sin(theta(t)), 0)

print(f"Pendulum equation: {eq}")

# Small angle approximation: sin(theta) ≈ theta
eq_linear = Eq(diff(theta(t), t, 2) + (g/L)*theta(t), 0)

# Solve linear equation
from sympy import dsolve
solution = dsolve(eq_linear, theta(t))
print(f"Small angle solution: {solution}")

# Harmonic oscillator
x = Function('x')
k = symbols('k', positive=True)

# mx'' + kx = 0
eq = Eq(m*diff(x(t), t, 2) + k*x(t), 0)
solution = dsolve(eq, x(t))
print(f"Harmonic oscillator: {solution}")

# Angular frequency
omega = sqrt(k/m)
print(f"Angular frequency: ω = {omega}")
```

### Quantum Mechanics

```python
from sympy import symbols, Function, diff, exp, sqrt, integrate
from sympy import pi, I, oo, simplify
from sympy import Rational

x, t = symbols('x t', real=True)
m, hbar, omega = symbols('m hbar omega', positive=True)

# Quantum harmonic oscillator wavefunction (ground state)
psi_0 = (m*omega/(pi*hbar))**Rational(1,4) * exp(-m*omega*x**2/(2*hbar))

print(f"Ground state: ψ₀(x) = {psi_0}")

# Verify normalization
norm = integrate(psi_0**2, (x, -oo, oo))
norm_simplified = simplify(norm)
print(f"Normalization: ∫|ψ₀|²dx = {norm_simplified}")

# Expectation value of x
expect_x = integrate(x * psi_0**2, (x, -oo, oo))
print(f"⟨x⟩ = {simplify(expect_x)}")

# Expectation value of x²
expect_x2 = integrate(x**2 * psi_0**2, (x, -oo, oo))
expect_x2_simplified = simplify(expect_x2)
print(f"⟨x²⟩ = {expect_x2_simplified}")

# Uncertainty
delta_x = sqrt(expect_x2_simplified)
print(f"Δx = {delta_x}")

# Momentum operator expectation
# ⟨p⟩ = -iℏ∫ ψ* (dψ/dx) dx
dpsi_dx = diff(psi_0, x)
expect_p = integrate(-I*hbar*psi_0*dpsi_dx, (x, -oo, oo))
print(f"⟨p⟩ = {simplify(expect_p)}")
```

### Electromagnetism

```python
from sympy import symbols, sin, cos, exp, sqrt, pi
from sympy import Function, diff, integrate
from sympy import I, oo

x, y, z, t = symbols('x y z t', real=True)
c, epsilon_0, mu_0 = symbols('c epsilon_0 mu_0', positive=True)

# Maxwell's equations in vacuum (symbolic)
E = Function('E')  # Electric field
B = Function('B')  # Magnetic field

# Gauss's law: ∇·E = 0 (in vacuum)
div_E = diff(E(x, y, z, t), x) + diff(E(x, y, z, t), y) + diff(E(x, y, z, t), z)

# Plane wave solution
k, omega = symbols('k omega', real=True)
E_0 = symbols('E_0', real=True)

# Electric field: E = E₀ exp(i(kx - ωt))
E_wave = E_0 * exp(I*(k*x - omega*t))
B_wave = E_wave / c  # Magnetic field

print(f"Electric field: E = {E_wave}")
print(f"Magnetic field: B = {B_wave}")

# Dispersion relation
dispersion = omega - c*k
print(f"Dispersion: ω = {c*k}")

# Poynting vector: S = (1/μ₀) E × B
# For plane wave in x-direction
S = E_0**2 / (mu_0 * c)
print(f"Time-averaged Poynting vector: ⟨S⟩ = {S}")

# Energy density
u = epsilon_0 * E_0**2 / 2
print(f"Energy density: u = {u}")
```

## Number Theory and Discrete Math

### Prime Numbers and Factorization

```python
from sympy import primefactors, factorint, isprime, prime
from sympy import nextprime, prevprime, primerange
from sympy import divisors, totient

# Check if prime
n = 17
print(f"{n} is prime: {isprime(n)}")

# Prime factorization
n = 360
factors = factorint(n)
print(f"Prime factorization of {n}: {factors}")
# {2: 3, 3: 2, 5: 1} means 2³ · 3² · 5

# List of prime factors
primes = primefactors(n)
print(f"Prime factors: {primes}")

# n-th prime
print(f"10th prime: {prime(10)}")  # 29

# Next/previous prime
p = 100
print(f"Next prime after {p}: {nextprime(p)}")
print(f"Previous prime before {p}: {prevprime(p)}")

# Primes in range
primes_list = list(primerange(1, 50))
print(f"Primes up to 50: {primes_list}")

# Divisors
divs = divisors(n)
print(f"Divisors of {n}: {divs}")

# Euler's totient function
phi = totient(n)
print(f"φ({n}) = {phi}")
```

### Modular Arithmetic

```python
from sympy import symbols, Mod, gcd, lcm
from sympy import mod_inverse, crt, isprime

# Modular arithmetic
a, b, m = symbols('a b m', integer=True)

# Basic modular operations
print(f"17 mod 5 = {Mod(17, 5)}")  # 2

# GCD and LCM
print(f"gcd(48, 18) = {gcd(48, 18)}")  # 6
print(f"lcm(48, 18) = {lcm(48, 18)}")  # 144

# Modular inverse
# Find x such that (a*x) mod m = 1
a, m = 3, 11
inv = mod_inverse(a, m)
print(f"{a}⁻¹ mod {m} = {inv}")
# Verify
print(f"Verification: {Mod(a * inv, m)}")  # Should be 1

# Chinese Remainder Theorem
# Solve: x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)
remainders = [2, 3, 2]
moduli = [3, 5, 7]
solution = crt(moduli, remainders)
print(f"CRT solution: x = {solution[0]} (mod {solution[1]})")

# Verify
x = solution[0]
for r, m in zip(remainders, moduli):
    print(f"x mod {m} = {Mod(x, m)} (should be {r})")
```

### Combinatorics

```python
from sympy import symbols, factorial, binomial, multinomial
from sympy import catalan, bell, fibonacci, lucas
from sympy import stirling, partition

n, k = symbols('n k', integer=True, positive=True)

# Factorials
print(f"10! = {factorial(10)}")

# Binomial coefficients
print(f"C(10, 3) = {binomial(10, 3)}")  # 120

# Multinomial coefficients
# (n choose k1, k2, ..., km) = n! / (k1! k2! ... km!)
print(f"Multinomial(5, [2, 2, 1]) = {multinomial(5, [2, 2, 1])}")

# Catalan numbers
for i in range(6):
    print(f"C_{i} = {catalan(i)}")

# Bell numbers (number of partitions of a set)
for i in range(6):
    print(f"B_{i} = {bell(i)}")

# Fibonacci and Lucas numbers
print(f"Fibonacci: {[fibonacci(i) for i in range(10)]}")
print(f"Lucas: {[lucas(i) for i in range(10)]}")

# Stirling numbers (first kind)
print(f"s(5, 2) = {stirling(5, 2, kind=1)}")

# Stirling numbers (second kind)
print(f"S(5, 2) = {stirling(5, 2, kind=2)}")

# Integer partitions
from sympy.utilities.iterables import partitions
p = list(partitions(5))
print(f"Partitions of 5: {p}")
```

## Practical Workflows

### Symbolic Regression Analysis

```python
from sympy import symbols, exp, log, sin, cos
from sympy import lambdify, simplify
import numpy as np
from scipy.optimize import curve_fit

# Generate synthetic data
x_data = np.linspace(0, 10, 50)
y_true = 2.5 * np.exp(-0.5 * x_data) * np.sin(x_data)
y_data = y_true + 0.1 * np.random.randn(len(x_data))

# Guess functional form
x = symbols('x')
A, alpha, omega = symbols('A alpha omega', real=True)

# Model: y = A * exp(-alpha*x) * sin(omega*x)
model = A * exp(-alpha*x) * sin(omega*x)

print(f"Model: y = {model}")

# Convert to numerical function
f_model = lambdify((x, A, alpha, omega), model, 'numpy')

# Fit using scipy
def fit_func(x_vals, A_val, alpha_val, omega_val):
    return f_model(x_vals, A_val, alpha_val, omega_val)

params, covariance = curve_fit(fit_func, x_data, y_data, p0=[2, 0.5, 1])

A_fit, alpha_fit, omega_fit = params
print(f"\nFitted parameters:")
print(f"A = {A_fit:.4f} (true: 2.5)")
print(f"α = {alpha_fit:.4f} (true: 0.5)")
print(f"ω = {omega_fit:.4f} (true: 1.0)")

# Create fitted expression
fitted_expr = model.subs([(A, A_fit), (alpha, alpha_fit), (omega, omega_fit)])
print(f"\nFitted expression: y = {fitted_expr}")

# Plot
import matplotlib.pyplot as plt

y_fit = f_model(x_data, A_fit, alpha_fit, omega_fit)

plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, alpha=0.5, label='Data')
plt.plot(x_data, y_true, 'g--', label='True')
plt.plot(x_data, y_fit, 'r-', label='Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('symbolic_fit.png', dpi=150)
```

### Optimization Problem with Constraints

```python
from sympy import symbols, diff, solve, simplify
from sympy import Matrix, hessian

# Optimization: minimize f(x,y) = x² + y²
# Subject to: g(x,y) = x + y - 1 = 0

x, y, lam = symbols('x y lambda', real=True)

# Objective function
f = x**2 + y**2

# Constraint
g = x + y - 1

# Lagrangian: L = f - λg
L = f - lam * g

# KKT conditions: ∇L = 0
dL_dx = diff(L, x)
dL_dy = diff(L, y)
dL_dlam = diff(L, lam)

print("KKT conditions:")
print(f"∂L/∂x = {dL_dx} = 0")
print(f"∂L/∂y = {dL_dy} = 0")
print(f"∂L/∂λ = {dL_dlam} = 0")

# Solve system
solution = solve([dL_dx, dL_dy, dL_dlam], [x, y, lam])
print(f"\nSolution: {solution}")

x_opt = solution[x]
y_opt = solution[y]

print(f"\nOptimal point: ({x_opt}, {y_opt})")
print(f"Objective value: f({x_opt}, {y_opt}) = {f.subs([(x, x_opt), (y, y_opt)])}")

# Verify constraint
g_check = g.subs([(x, x_opt), (y, y_opt)])
print(f"Constraint check: g = {simplify(g_check)}")

# Second-order conditions (Hessian)
H = hessian(f, (x, y))
print(f"\nHessian of f:\n{H}")

# Evaluate at optimum
H_opt = H.subs([(x, x_opt), (y, y_opt)])
eigenvals = H_opt.eigenvals()
print(f"Eigenvalues at optimum: {eigenvals}")
if all(val > 0 for val in eigenvals.keys()):
    print("✓ Confirmed: Local minimum")
```

### Symbolic Differential Equation Solver

```python
from sympy import symbols, Function, dsolve, Eq, diff
from sympy import sin, cos, exp, sqrt
from sympy import init_printing

init_printing()

x = symbols('x')
y = Function('y')

# Collection of ODEs to solve
odes = [
    # Linear first-order
    (Eq(diff(y(x), x) + y(x), exp(x)), "Linear first-order"),
    
    # Separable
    (Eq(diff(y(x), x), y(x) * (1 - y(x))), "Logistic equation"),
    
    # Second-order linear
    (Eq(diff(y(x), x, 2) - 4*y(x), 0), "Simple harmonic"),
    
    # Damped oscillator
    (Eq(diff(y(x), x, 2) + 2*diff(y(x), x) + 5*y(x), 0), "Damped oscillator"),
]

print("SOLVING DIFFERENTIAL EQUATIONS")
print("=" * 60)

for eq, description in odes:
    print(f"\n{description}:")
    print(f"Equation: {eq}")
    
    # Solve
    solution = dsolve(eq, y(x))
    print(f"Solution: {solution}")
    
    # Verify solution
    y_sol = solution.rhs
    eq_check = eq.lhs.subs(y(x), y_sol)
    eq_check_simplified = simplify(eq_check)
    
    if eq_check_simplified == eq.rhs:
        print("✓ Solution verified")
    else:
        print(f"Verification: {eq_check_simplified} = {eq.rhs}")

# Solve with initial conditions
print("\n" + "=" * 60)
print("WITH INITIAL CONDITIONS")
print("=" * 60)

# y' + y = 0, y(0) = 1
eq = Eq(diff(y(x), x) + y(x), 0)
ics = {y(0): 1}

solution = dsolve(eq, y(x), ics=ics)
print(f"Equation: {eq}, y(0) = 1")
print(f"Solution: {solution}")
```

### Taylor Series Approximation Analysis

```python
from sympy import symbols, sin, cos, exp, log, sqrt
from sympy import series, diff, factorial
from sympy import lambdify, Abs
import numpy as np
import matplotlib.pyplot as plt

x = symbols('x')

# Function to approximate
f = exp(sin(x))

# Calculate Taylor series of different orders
orders = [1, 3, 5, 7, 9]
series_dict = {}

print(f"Taylor series for f(x) = {f} around x=0:")
print("=" * 60)

for n in orders:
    s = series(f, x, 0, n)
    s_poly = s.removeO()
    series_dict[n] = s_poly
    print(f"Order {n-1}: {s}")

# Convert to numerical functions
f_num = lambdify(x, f, 'numpy')
series_num = {n: lambdify(x, s, 'numpy') for n, s in series_dict.items()}

# Evaluate
x_vals = np.linspace(-2, 2, 200)
y_true = f_num(x_vals)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Series approximations
ax1.plot(x_vals, y_true, 'k-', linewidth=2, label='True function')
for n in orders:
    y_approx = series_num[n](x_vals)
    ax1.plot(x_vals, y_approx, '--', label=f'Order {n-1}', alpha=0.7)

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Taylor Series Approximations')
ax1.legend()
ax1.grid(True)
ax1.set_ylim(-2, 5)

# Error analysis
for n in orders:
    y_approx = series_num[n](x_vals)
    error = np.abs(y_true - y_approx)
    ax2.semilogy(x_vals, error, label=f'Order {n-1}')

ax2.set_xlabel('x')
ax2.set_ylabel('Absolute Error')
ax2.set_title('Approximation Error')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('taylor_series_analysis.png', dpi=150)

# Print error at specific points
print("\n" + "=" * 60)
print("ERROR AT x = 1:")
print("=" * 60)
x_test = 1.0
y_test = f_num(x_test)

for n in orders:
    y_approx = series_num[n](x_test)
    error = abs(y_test - y_approx)
    rel_error = error / abs(y_test) * 100
    print(f"Order {n-1}: Error = {error:.6e} ({rel_error:.2f}%)")
```

### Symbolic Integration Techniques

```python
from sympy import symbols, integrate, sin, cos, exp, log, sqrt
from sympy import apart, trigsimp, simplify
from sympy import pi, oo, I

x = symbols('x')

# Collection of integrals demonstrating different techniques
integrals = [
    # Substitution (automatic)
    (x * exp(x**2), "u-substitution (automatic)"),
    
    # Integration by parts (automatic)
    (x * sin(x), "Integration by parts"),
    
    # Trigonometric
    (sin(x)**2, "Trigonometric identity"),
    
    # Rational functions (partial fractions)
    (1/(x**2 + 3*x + 2), "Partial fractions"),
    
    # Logarithmic
    (log(x)/x, "Logarithmic"),
    
    # Radical
    (1/sqrt(1 - x**2), "Inverse trig"),
]

print("INTEGRATION TECHNIQUES")
print("=" * 60)

for integrand, technique in integrals:
    print(f"\n{technique}:")
    print(f"∫ {integrand} dx")
    
    # Compute integral
    result = integrate(integrand, x)
    print(f"= {result}")
    
    # Verify by differentiation
    check = diff(result, x)
    check_simplified = simplify(check)
    
    if simplify(check_simplified - integrand) == 0:
        print("✓ Verified by differentiation")
    else:
        print(f"Check: d/dx({result}) = {check_simplified}")

# Definite integrals
print("\n" + "=" * 60)
print("DEFINITE INTEGRALS")
print("=" * 60)

definite_integrals = [
    (x**2, 0, 1, "∫₀¹ x² dx"),
    (sin(x), 0, pi, "∫₀^π sin(x) dx"),
    (exp(-x**2), -oo, oo, "∫₋∞^∞ e^(-x²) dx"),
    (1/(1 + x**2), -oo, oo, "∫₋∞^∞ 1/(1+x²) dx"),
]

for integrand, a, b, description in definite_integrals:
    result = integrate(integrand, (x, a, b))
    print(f"{description} = {result}")
    if result.is_number:
        print(f"  ≈ {result.evalf()}")
```

This comprehensive SymPy guide covers 60+ examples across symbolic mathematics, calculus, algebra, and more!
