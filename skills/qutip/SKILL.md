---
name: qutip
description: Quantum Toolbox in Python. Framework for simulating the dynamics of open quantum systems. Provides data structures for quantum objects (kets, bras, operators) and solvers for master equations, Monte Carlo trajectories, and time-dependent Hamiltonians. Use for quantum dynamics simulation, open quantum systems, master equations, quantum optics, cavity QED, Jaynes-Cummings model, Rabi oscillations, Wigner functions, quantum correlations, entanglement analysis, and quantum control.
version: 5.0
license: BSD-3-Clause
---

# QuTiP - Quantum Dynamics

QuTiP is designed to be a flexible and efficient framework for quantum mechanics. It allows for easy creation of quantum objects and provides powerful solvers to track their evolution in both closed and open environments.

## When to Use

- Simulating the time evolution of a quantum system (Schrödinger or Master equation)
- Calculating steady states of open quantum systems (e.g., a cavity under drive and dissipation)
- Analyzing entanglement, Wigner functions, and quantum correlations
- Studying light-matter interactions (Jaynes-Cummings model, Rabi oscillations)
- Pulse sequence optimization and quantum control
- Calculating spectrums and multi-time correlation functions
- Parallelizing quantum simulations across multiple CPUs

## Reference Documentation

**Official docs**: https://qutip.org/docs/latest/  
**Tutorials**: https://qutip.org/tutorials.html  
**Search patterns**: `qutip.Qobj`, `qutip.mesolve`, `qutip.wigner`, `qutip.expect`, `qutip.basis`

## Core Principles

### The Qobj (Quantum Object)

The central data structure. A Qobj can represent a state (ket or bra), an operator, or a superoperator. It automatically handles dimensions and validates operations (e.g., preventing the addition of a ket and an operator).

### Composite Systems

QuTiP uses the Kronecker product (tensor) to represent systems composed of multiple sub-systems (e.g., a qubit coupled to a resonator).

### Solver Workflow

1. Define the Hamiltonian (H)
2. Define collapse operators (C_n) for dissipation
3. Set initial state (ρ₀)
4. Define time sequence (t)
5. Run the solver (mesolve, sesolve, etc.)

## Quick Reference

### Installation

```bash
pip install qutip
```

### Standard Imports

```python
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
```

### Basic Pattern - Rabi Oscillations

```python
import qutip as qt
import numpy as np

# 1. Define Hamiltonian: H = h_bar * omega * sigma_x / 2
omega = 1.0 * 2 * np.pi
H = 0.5 * omega * qt.sigmax()

# 2. Initial state: Spin down (|1>)
psi0 = qt.basis(2, 1)

# 3. Time points
times = np.linspace(0.0, 10.0, 100)

# 4. Solve Schrödinger Equation
result = qt.sesolve(H, psi0, times, [qt.sigmaz()])

# 5. Get expectation values
sz_expt = result.expect[0]
```

## Critical Rules

### ✅ DO

- **Use basis(N, i)** - Use built-in functions to create states instead of manual arrays
- **Check Qobj Dimensions** - Always verify `.dims` when working with composite systems to ensure tensor products are correct
- **Use expect()** - Let solvers calculate expectation values directly to save memory and time
- **Specify c_ops for Open Systems** - Use mesolve with collapse operators to model decoherence
- **Vectorize Hamiltonians** - If your Hamiltonian is time-dependent, use the `[H0, [H1, 'sin(w*t)']]` list format for speed
- **Visualize with Wigner** - Use `qt.wigner(state, xvec, yvec)` for a deep look into the quantum nature of the state

### ❌ DON'T

- **Extract data manually** - Avoid `state.data.toarray()` unless absolutely necessary; QuTiP's Qobj is optimized for sparse math
- **Ignore solver warnings** - Numerical instabilities often mean the time step is too large or the Hilbert space is truncated too early
- **Mix Hilbert Spaces** - Don't perform operations between objects with different dimensions without proper tensor products
- **Forget Truncation** - For infinite systems (like oscillators), ensure the Fock space size (N) is large enough for convergence

## Anti-Patterns (NEVER)

```python
import qutip as qt

# ❌ BAD: Creating states using raw numpy arrays (no metadata)
psi = qt.Qobj([[1], [0]])

# ✅ GOOD: Use built-in constructors
psi = qt.basis(2, 0) # Ket |0>

# ❌ BAD: Manual Kronecker product
H_total = np.kron(H_atom, np.eye(N_cav))

# ✅ GOOD: Use qt.tensor
H_total = qt.tensor(H_atom, qt.qeye(N_cav))

# ❌ BAD: Calculating expectation values in a Python loop
# (Extremely slow and memory intensive)

# ✅ GOOD: Pass operators to the solver
result = qt.mesolve(H, psi0, times, c_ops, [qt.sigmaz(), qt.sigmax()])
```

## Quantum Objects (qt.Qobj)

### Creation and Properties

```python
# Kets and Bras
psi = qt.basis(5, 0) # Ket |0> in 5-level system
bra = psi.dag()      # Adjoint (Bra)

# Operators
a = qt.destroy(10) # Annihilation operator (size 10)
n = a.dag() * a    # Number operator
sz = qt.sigmaz()   # Pauli Z

# Properties
print(psi.type)    # 'ket'
print(sz.dims)     # [[2], [2]]
print(n.eigenstates()) # Get eigenvalues and vectors
```

## Composite Systems (Tensors)

```python
# Qubit (2) + Cavity (10)
sz = qt.sigmaz()
a = qt.destroy(10)

# Operator acting on qubit only
sz_total = qt.tensor(sz, qt.qeye(10))

# Interaction term: sigma_x * (a + a_dag)
H_int = qt.tensor(qt.sigmax(), a + a.dag())
```

## Dynamics Solvers (qt.mesolve, qt.sesolve)

### Master Equation (Open Systems)

```python
# Parameters
kappa = 0.1 # Cavity decay rate
H = qt.tensor(qt.sigmax(), qt.qeye(10)) # Hamiltonian
c_ops = [np.sqrt(kappa) * qt.tensor(qt.qeye(2), a)] # Decay list
psi0 = qt.tensor(qt.basis(2, 0), qt.basis(10, 0))
times = np.linspace(0, 50, 200)

# Solve
result = qt.mesolve(H, psi0, times, c_ops, [qt.tensor(qt.sigmaz(), qt.qeye(10))])

# Access expectation values
exp_z = result.expect[0]
```

### Time-Dependent Hamiltonians

```python
# Using string-based syntax (fastest)
H0 = qt.sigmax()
H1 = qt.sigmaz()
args = {'w': 0.5}
H_t = [H0, [H1, 'sin(w * t)']]

result = qt.sesolve(H_t, qt.basis(2, 0), times, args=args)
```

### Steady State and Correlation

```python
# Steady state of a driven-dissipative system
rho_ss = qt.steadystate(H, c_ops)

# Expectation value in steady state
avg_n = qt.expect(a.dag() * a, rho_ss)

# Emission Spectrum
w_list = np.linspace(0, 10, 100)
spec = qt.spectrum(H, w_list, c_ops, a.dag(), a)
```

## Visualization

### Wigner and Bloch Sphere

```python
# Wigner function for a Cat state
cat = (qt.coherent(20, 2) + qt.coherent(20, -2)).unit()
xvec = np.linspace(-5, 5, 100)
W = qt.wigner(cat, xvec, xvec)

# Bloch Sphere
b = qt.Bloch()
b.add_states(qt.basis(2, 0))
b.add_states((qt.basis(2,0) + qt.basis(2,1)).unit())
# b.show()
```

## Practical Workflows

### 1. Jaynes-Cummings Model (Qubit-Cavity)

```python
def simulate_jc(g, kappa, n_max, times):
    # Operators
    a = qt.tensor(qt.qeye(2), qt.destroy(n_max))
    sm = qt.tensor(qt.destroy(2), qt.qeye(n_max))
    
    # Hamiltonian
    H = g * (sm.dag() * a + sm * a.dag())
    
    # Dissipation
    c_ops = [np.sqrt(kappa) * a]
    
    # Initial: Excited atom, 0 photons
    psi0 = qt.tensor(qt.basis(2, 0), qt.basis(n_max, 0))
    
    res = qt.mesolve(H, psi0, times, c_ops, [sm.dag()*sm, a.dag()*a])
    return res.expect
```

### 2. Quantum State Tomography Analysis

```python
def fidelity_check(state_ideal, state_measured):
    """Calculates fidelity between two quantum states."""
    # Works for both kets and density matrices
    return qt.fidelity(state_ideal, state_measured)
```

### 3. Calculating G2 Correlation Function

```python
def calculate_g2(H, c_ops, a_op, times):
    """Second-order correlation function g2(tau)."""
    # Use built-in correlation function
    g2_tau = qt.correlation_2op_1t(H, None, times, c_ops, a_op.dag(), a_op)
    # Normalize by steady state intensity
    rho_ss = qt.steadystate(H, c_ops)
    n_ss = qt.expect(a_op.dag() * a_op, rho_ss)
    return np.real(g2_tau) / (n_ss**2)
```

## Performance Optimization

### Sparse vs Dense

QuTiP uses Scipy sparse matrices by default. For very small systems (N < 4), converting to dense might be faster, but for most N > 10, sparse is mandatory.

### Parallel Map

Use `qt.parallel_map` for parameter sweeps.

```python
def task(omega):
    H = 0.5 * omega * qt.sigmax()
    return qt.mesolve(H, psi0, times, c_ops, [qt.sigmaz()]).expect[0]

omegas = np.linspace(0, 10, 20)
results = qt.parallel_map(task, omegas)
```

## Common Pitfalls and Solutions

### The "Dimensions Mismatch" Error

When tensoring systems, the order must be consistent.

```python
# ❌ Problem: H is (2x10) but Op is (10x2)
# ✅ Solution: Check dims and ensure consistent order in tensor()
print(H.dims) # [[2, 10], [2, 10]]
```

### Insufficient Fock Space

If your photon number distribution hits the upper limit of your N, your results are invalid.

```python
# ✅ Solution: Check populations
occ = qt.expect(qt.destroy(N).dag() * qt.destroy(N), final_state)
# If occ > 0.8 * N, increase N.
```

### Hermitian Hamiltonian

Solvers assume H is Hermitian. If your time-dependent code results in a non-Hermitian H, sesolve will produce unphysical states (norm ≠ 1).

```python
# ✅ Solution: Check norm
if not np.allclose(result.states[-1].norm(), 1.0):
    print("Warning: Norm not preserved!")
```

## Best Practices

1. Always use built-in constructors (`basis`, `coherent`, etc.) instead of manual array creation
2. Verify dimensions with `.dims` before performing tensor products
3. Pass operators to solvers for expectation values rather than calculating them manually
4. Use appropriate Fock space truncation (N) - check that populations don't saturate
5. For time-dependent Hamiltonians, use the string-based format for best performance
6. Monitor solver warnings - they often indicate numerical issues
7. Use sparse matrices (default) for systems with N > 10
8. Check state normalization after time evolution, especially for custom solvers
9. Use `parallel_map` for parameter sweeps to leverage multiple CPUs
10. Visualize quantum states with Wigner functions and Bloch spheres for deeper insight

QuTiP is the standard tool for exploring the non-intuitive world of quantum dynamics. It bridges the gap between theoretical equations and numerical experimentation, providing physicists with a high-performance lab in a Python script.
