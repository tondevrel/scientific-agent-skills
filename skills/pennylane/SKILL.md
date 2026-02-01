---
name: pennylane
description: Cross-platform Python library for differentiable quantum computing. Integrated with machine learning libraries like PyTorch, TensorFlow, and JAX. Designed for quantum machine learning (QML), variational algorithms, and hardware-agnostic quantum programming. Use for Quantum Neural Networks (QNNs), Variational Quantum Algorithms (VQE, QAOA), hybrid classical-quantum machine learning, quantum chemistry calculations, benchmarking quantum algorithms, optimizing quantum control pulses, and investigating QML phenomena like Barren Plateaus.
version: 0.35
license: Apache-2.0
---

# PennyLane - Quantum Machine Learning

PennyLane treats quantum computers like neural network layers. It allows for the calculation of gradients of quantum circuits (using the parameter-shift rule or backpropagation), enabling the optimization of hybrid classical-quantum models.

## When to Use

- Developing and training Quantum Neural Networks (QNNs)
- Variational Quantum Algorithms (VQE, QAOA)
- Hybrid classical-quantum machine learning (e.g., Quantum CNNs)
- Quantum chemistry calculations in a differentiable framework
- Benchmarking quantum algorithms across different hardware (IBM, Rigetti, Xanadu, IonQ)
- Optimizing quantum control pulses
- Investigating Barren Plateaus and other QML-specific phenomena

## Reference Documentation

**Official docs**: https://docs.pennylane.ai/  
**Demos/Tutorials**: https://pennylane.ai/qml/demonstrations.html  
**Search patterns**: `qml.qnode`, `qml.device`, `qml.expval`, `qml.grad`, `qml.templates`

## Core Principles

### The QNode (Quantum Node)

A QNode is a quantum circuit bound to a device, which can be called like a standard Python function. It is the fundamental unit that PennyLane can differentiate.

### Hardware Agnosticism

PennyLane provides a unified interface. The same code can run on a high-performance simulator (default.qubit), a GPU-accelerated backend (lightning.qubit), or real quantum hardware.

### Automatic Differentiation

Quantum circuits in PennyLane are "aware" of their gradients. You can use standard optimizers (Adam, SGD) to tune rotation angles in the circuit.

## Quick Reference

### Installation

```bash
pip install pennylane
# For GPU support
pip install pennylane-lightning[gpu]
```

### Standard Imports

```python
import pennylane as qml
from pennylane import numpy as np # Use PennyLane's wrapped NumPy for gradients
```

### Basic Pattern - Differentiable Circuit

```python
import pennylane as qml
from pennylane import numpy as np

# 1. Define Device
dev = qml.device("default.qubit", wires=2)

# 2. Define QNode (Quantum Node)
@qml.qnode(dev)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(1))

# 3. Calculate Gradient
params = np.array([0.1, 0.2], requires_grad=True)
grad_fn = qml.grad(circuit)
print(f"Gradient: {grad_fn(params)}")
```

## Critical Rules

### ✅ DO

- **Use qml.numpy** - Always use the PennyLane-wrapped NumPy for parameters you want to differentiate
- **Use Templates** - Instead of building layers manually, use `qml.templates` (like `StronglyEntanglingLayers`) for efficient QML architectures
- **Set requires_grad** - Explicitly mark trainable parameters with `requires_grad=True`
- **Prefer lightning.qubit** - For simulations with >15 qubits, use the lightning device for significantly better performance
- **Use Broadcasting** - Many gates support broadcasting over input arrays, which is faster than loops
- **Batch Circuits** - Use `qml.batch_input` or `qml.map` for processing multiple inputs simultaneously

### ❌ DON'T

- **Mix NumPy versions** - Using standard `import numpy as np` for trainable parameters will break the autograd engine
- **Overcomplicate Ansätze** - Start with simple circuits to avoid Barren Plateaus (where gradients vanish)
- **Use too many qubits on simulators** - Memory scales as 2^N. 30 qubits require ~16GB of RAM for a single statevector
- **Hardcode Wire Indices** - Use variables for wires to make your code reusable and scalable

## Anti-Patterns (NEVER)

```python
import pennylane as qml
# ❌ BAD: Standard numpy for trainable parameters
import numpy as np 

# ✅ GOOD: PennyLane's wrapped numpy
from pennylane import numpy as np

# ❌ BAD: Creating a device inside a loop
for i in range(100):
    dev = qml.device("default.qubit", wires=2) # Expensive initialization

# ✅ GOOD: Define device once
dev = qml.device("default.qubit", wires=2)

# ❌ BAD: Manual parameter shifting
# (shift = 0.5 * pi, calc f(x+s) - f(x-s)...)
# ✅ GOOD: Let PennyLane handle it automatically
grad = qml.grad(circuit)(params)
```

## Circuit Templates (qml.templates)

### Built-in QML Layers

```python
import pennylane as qml

@qml.qnode(dev)
def qnn_layer(inputs, weights):
    # Encoding classical data into quantum state
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    
    # Trainable entangling layer
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    return qml.expval(qml.PauliZ(0))
```

## Hybrid Optimization (PyTorch/TF)

### Integration with Classical Frameworks

```python
import torch
import pennylane as qml

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev, interface="torch")
def quantum_layer(phi, theta):
    qml.RX(phi, wires=0)
    qml.RZ(theta, wires=1)
    return qml.expval(qml.PauliZ(0))

# Now this QNode can be used in a Torch Module
phi = torch.tensor(0.1, requires_grad=True)
theta = torch.tensor(0.2, requires_grad=True)
result = quantum_layer(phi, theta)
result.backward()
print(phi.grad)
```

## Quantum Chemistry (qml.qchem)

### VQE Workflow

```python
from pennylane import qchem

# 1. Define molecule
symbols = ["H", "H"]
coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])

# 2. Build Hamiltonian
H, qubits = qchem.molecular_hamiltonian(symbols, coordinates)

# 3. Define VQE circuit
dev = qml.device("default.qubit", wires=qubits)
@qml.qnode(dev)
def vqe_circuit(params):
    # Simple Hartree-Fock state + rotations
    qml.BasisState(np.array([1, 1, 0, 0]), wires=range(qubits))
    qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
    return qml.expval(H)
```

## Measurements and Observables

### Beyond Expectation Values

```python
@qml.qnode(dev)
def multi_measure(params):
    qml.RX(params[0], wires=0)
    return (
        qml.expval(qml.PauliZ(0)), # Expectation value
        qml.var(qml.PauliZ(0)),    # Variance
        qml.probs(wires=[0, 1]),   # Probabilities
        qml.sample(qml.PauliX(1))  # Raw samples (shots)
    )
```

## Practical Workflows

### 1. Quantum Variational Classifier

```python
def variational_classifier(weights, bias, x):
    """A simple QNN classifier."""
    return circuit(weights, x) + bias

def cost(weights, bias, X, Y):
    predictions = [variational_classifier(weights, bias, x) for x in X]
    return square_loss(Y, predictions)

# Optimizer
opt = qml.AdamOptimizer(stepsize=0.1)
# ... loop with opt.step(cost, ...)
```

### 2. Computing Gradients on Hardware (Parameter-Shift)

```python
# When running on real hardware, PennyLane automatically uses
# the parameter-shift rule to calculate gradients via multiple executions.
dev_remote = qml.device("braket.aws.qubit", device_arn="...", wires=2)

@qml.qnode(dev_remote, diff_method="parameter-shift")
def hardware_circuit(params):
    qml.RY(params[0], wires=0)
    return qml.expval(qml.PauliZ(0))
```

### 3. Noise Simulation

```python
dev_noisy = qml.device("default.mixed", wires=2)

@qml.qnode(dev_noisy)
def noisy_circuit(p):
    qml.Hadamard(wires=0)
    # Apply depolarizing noise to wire 0
    qml.DepolarizingChannel(p, wires=0)
    return qml.state()
```

## Performance Optimization

### JAX Integration for Speed

JAX is often the fastest interface for large-scale simulations due to its Just-In-Time (JIT) compilation.

```python
import jax

@qml.qnode(dev, interface="jax")
def fast_circuit(x):
    qml.RX(x, wires=0)
    return qml.expval(qml.PauliZ(0))

jit_circuit = jax.jit(fast_circuit)
```

### Adjoint Differentiation

For large circuits, `diff_method="adjoint"` is much more memory-efficient than backpropagation.

```python
dev = qml.device("lightning.qubit", wires=20)
@qml.qnode(dev, diff_method="adjoint")
def large_circuit(params):
    ...
```

## Common Pitfalls and Solutions

### The "Vanishing Gradient" (Barren Plateaus)

As the number of qubits and layers increases, the gradient often becomes exponentially small.

```python
# ✅ Solution: 
# 1. Use better initialization for weights.
# 2. Use local observables (PauliZ(i)) instead of global ones.
# 3. Use identity-block initialization.
```

### Non-Trainable Inputs

Sometimes you want to pass data (like images) that shouldn't be optimized.

```python
# ✅ Solution: Use the 'argnum' in qml.grad or use non-array types
# Or explicitly:
params = np.array(0.1, requires_grad=True)
data = np.array(0.5, requires_grad=False)
```

### Output Shape Mismatch

`qml.probs` returns an array of size 2^N.

```python
# ❌ Problem: Using probs(wires=[0,1,2]) in a loss function expecting a scalar.
# ✅ Solution: Use expval() for a single scalar or handle the distribution.
```

PennyLane bridges the gap between quantum physics and artificial intelligence. By making quantum circuits differentiable, it transforms them into powerful, trainable tools for the next generation of scientific computing.
