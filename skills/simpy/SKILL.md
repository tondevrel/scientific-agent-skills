---
name: simpy
description: A process-based discrete-event simulation framework. Use for modeling queuing systems, supply chains, manufacturing processes, network simulation, project management, and any system where events occur at specific points in time. Load when working with discrete event simulation, process modeling, resource allocation, virtual time, simpy.Environment, simpy.Resource, or event-driven simulation.
version: 4.1
license: MIT
---

# SimPy - Discrete Event Simulation

SimPy allows you to model real-world processes as Python generators. Components in SimPy (like customers, cars, or data packets) are "processes" that interact with each other and with limited "resources" (like servers, parking spots, or bandwidth).

## When to Use

- **Modeling queuing systems** (Bank tellers, call centers, hospital emergency rooms)
- **Simulating supply chains and logistics** (Warehouses, transport networks)
- **Analyzing manufacturing processes** (Assembly lines, machine maintenance)
- **Network simulation** (Packet routing, server load balancing)
- **Project management** (Task dependencies, resource allocation)
- **Any system where "events" occur at specific points in time rather than continuously**

## Reference Documentation

**Official docs**: https://simpy.readthedocs.io/  
**GitHub**: https://github.com/simpy/simpy  
**Search patterns**: `simpy.Environment`, `simpy.Resource`, `env.process`, `yield env.timeout`

## Core Principles

### Virtual Time

SimPy uses an internal clock starting at 0. Time moves forward only when an event is processed. Between events, "nothing" happens, so simulating 100 years takes seconds if only a few events occur.

### Processes as Generators

Processes are standard Python functions using `yield`. When a process yields an event, SimPy suspends it until that event occurs.

### Resources

Three types of shared objects:
- **Resource**: Limited number of slots (e.g., a counter)
- **Container**: For bulk matter (e.g., a gas tank, RAM)
- **Store**: For distinct objects (e.g., a buffer of messages)

## Quick Reference

### Installation

```bash
pip install simpy
```

### Standard Imports

```python
import simpy
import random
```

### Basic Pattern - A Simple Process

```python
import simpy

def clock(env, name, tick):
    while True:
        print(f"{name} at {env.now}")
        # 'yield' tells SimPy to wait for this event
        yield env.timeout(tick)

# 1. Create Environment
env = simpy.Environment()

# 2. Add Process
env.process(clock(env, 'Fast', 0.5))
env.process(clock(env, 'Slow', 1.0))

# 3. Run for 2 units of virtual time
env.run(until=2.1)
```

## Critical Rules

### ✅ DO

- **Use `yield env.timeout(n)`** - This is the only way to advance time for a process. Never use `time.sleep()`.
- **Use `with resource.request() as req:`** - This ensures a resource is automatically released even if the process is interrupted or fails.
- **Set a random seed** - Always use `random.seed(42)` for reproducible simulations.
- **Pass the `env` object** - Every process and resource needs a reference to the environment to track time.
- **Use `env.run(until=...)`** - Always define a stopping condition, otherwise a `while True` simulation will run forever.
- **Collect data in lists** - Store timestamps and event details in external lists for post-simulation analysis (e.g., with Pandas).

### ❌ DON'T

- **Use real-world time** - simpy is for virtual time. `env.now` is your only clock.
- **Modify the environment directly** - Interact with it only through processes and events.
- **Forget to yield** - If you don't yield an event inside a loop, you create an infinite loop at the same instant in virtual time, hanging the simulation.
- **Block the event loop** - Heavy CPU calculations inside a process should be rare; they "stop" the virtual clock of the entire simulation while the CPU works.

## Anti-Patterns (NEVER)

```python
import simpy
import time

# ❌ BAD: Using real sleep
def bad_process(env):
    time.sleep(1) # This stops the REAL clock, but virtual time doesn't move!
    yield env.timeout(1)

# ✅ GOOD: Use virtual timeout
def good_process(env):
    yield env.timeout(1) # Moves virtual clock forward instantly

# ❌ BAD: Infinite loop without yielding
# def hang_forever(env):
#     while True:
#         x = 1 + 1 # No yield = simulation clock stuck at current env.now

# ❌ BAD: Manual resource release (Dangerous)
def risky_request(env, res):
    req = res.request()
    yield req
    # If something fails here, the resource is never released
    res.release(req)

# ✅ GOOD: Using context manager
def safe_request(env, res):
    with res.request() as req:
        yield req
        # Resource automatically released when exiting 'with'
```

## Resources and Queues (simpy.Resource)

### Managing Shared Capacity

```python
def car(env, name, bcs, driving_time, charge_time):
    # Simulate driving to the station
    yield env.timeout(driving_time)

    print(f'{name} arriving at {env.now}')
    # Request one of the charging spots
    with bcs.request() as req:
        yield req
        print(f'{name} starting to charge at {env.now}')
        yield env.timeout(charge_time)
        print(f'{name} leaving at {env.now}')

env = simpy.Environment()
# Battery Charging Station with 2 spots
bcs = simpy.Resource(env, capacity=2)

for i in range(4):
    env.process(car(env, f'Car {i}', bcs, i*2, 5))

env.run()
```

## Advanced Resources: Containers and Stores

### Containers (Liquids/Bulk)

```python
# Modeling a gas station tank
gas_tank = simpy.Container(env, capacity=1000, init=500)

def refuel(env, gas_tank):
    yield gas_tank.put(400) # Add fuel

def car(env, gas_tank):
    yield gas_tank.get(40) # Consume fuel
```

### Stores (Distinct Objects)

```python
# Modeling a producer-consumer buffer
buffer = simpy.Store(env, capacity=10)

def producer(env, buffer):
    yield buffer.put(f'Widget at {env.now}')

def consumer(env, buffer):
    item = yield buffer.get()
    print(f'Consumed {item}')
```

## Event Synchronization

### Waiting for Multiple Events

```python
# Wait for BOTH events
yield env.timeout(10) & other_process_event

# Wait for EITHER event
yield env.timeout(10) | other_process_event
```

## Practical Workflows

### 1. Hospital Emergency Room Model

```python
def patient(env, name, nurse, doctor):
    arrival_time = env.now
    
    # 1. Wait for Nurse (Triage)
    with nurse.request() as req:
        yield req
        yield env.timeout(random.expovariate(1/5)) # 5 mins avg
        
    # 2. Wait for Doctor (Treatment)
    with doctor.request() as req:
        yield req
        yield env.timeout(random.normalvariate(20, 5)) # 20 mins avg
        
    wait_times.append(env.now - arrival_time)

wait_times = []
env = simpy.Environment()
nurse = simpy.Resource(env, capacity=2)
doctor = simpy.Resource(env, capacity=1)
# ... launch processes ...
```

### 2. Machine Failure and Repair

```python
def machine(env, repairman):
    while True:
        try:
            # Working phase
            working_time = 100
            yield env.timeout(working_time)
        except simpy.Interrupt:
            # Interrupted by failure
            with repairman.request() as req:
                yield req
                yield env.timeout(10) # Repair time

def break_machine(env, machine_proc):
    while True:
        yield env.timeout(random.uniform(50, 200))
        machine_proc.interrupt()
```

### 3. Server Request Handling (Data Collection)

```python
import pandas as pd

log = []

def request(env, server):
    start_wait = env.now
    with server.request() as req:
        yield req
        wait = env.now - start_wait
        service_time = random.expovariate(1.0)
        yield env.timeout(service_time)
        
        log.append({
            'arrival': start_wait,
            'wait': wait,
            'service': service_time,
            'finish': env.now
        })

# After simulation:
# df = pd.DataFrame(log)
# print(df['wait'].mean())
```

## Performance Optimization

### Using simpy.PriorityResource

In systems where some events are more important (e.g., critical patients), use `PriorityResource`.

```python
res = simpy.PriorityResource(env, capacity=1)
with res.request(priority=0) as req: # Lower number = higher priority
    yield req
```

### Monitoring Large Simulations

Avoid printing to the console in high-speed simulations. Instead, use a data collection list and process it with NumPy/Pandas after the `env.run()` finishes.

## Common Pitfalls and Solutions

### The "Stuck Process"

If your code reaches a logic branch where no `yield` occurs in a `while True` loop, SimPy freezes.

```python
# ❌ Problem:
def logic_error(env):
    while True:
        if condition:
            yield env.timeout(1)
        # Else? If condition is false, we loop forever at same time point!

# ✅ Solution:
def logic_fixed(env):
    while True:
        if condition:
            yield env.timeout(1)
        else:
            yield env.timeout(0.1) # Or some default wait
```

### Multiple Environments

Never mix objects (Resources, Processes) from different environments. Every component must belong to the same `simpy.Environment()` instance.

### Interruption Timing

When interrupting a process, the interrupt happens at the current virtual time. Be careful with what the interrupted process was yielding (it will receive a `simpy.Interrupt` exception).

## Best Practices

1. **Always use context managers for resources** - `with resource.request() as req:` ensures proper cleanup
2. **Set random seeds for reproducibility** - Use `random.seed(42)` at the start of simulations
3. **Collect data externally** - Store event data in lists/dictionaries, not just print statements
4. **Define clear stopping conditions** - Always use `env.run(until=...)` with a specific time limit
5. **Use virtual time only** - Never mix `time.sleep()` or real-world time with SimPy
6. **Test with small simulations first** - Verify logic before running large-scale simulations
7. **Document your processes** - Clearly name processes and resources for debugging
8. **Handle interruptions properly** - Use try/except blocks when processes can be interrupted

SimPy turns Python's generator syntax into a powerful engine for modeling time and scarcity. It allows scientists and engineers to experiment with "What if?" scenarios for complex systems without the cost or risk of physical trials.
