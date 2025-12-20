# Bee-Population-Modeling

# DyBePo: Dynamic Beehive Population Model

## Overview
**DyBePo** is a Python-based mathematical simulation of honeybee (*Apis mellifera*) colony dynamics. Based on the research paper *"DyBePo: Dynamic Beehive Population Modeling"*, this project uses a system of Ordinary Differential Equations (ODEs) to model the complex feedback loops within a hive, including age polyethism, social inhibition, resource management, and seasonal fluctuations.

The model explicitly tracks six stages of bees and one resource variable:
* **E**: Eggs
* **L**: Larvae
* **P**: Pupae
* **D**: Drones
* **H**: Hive Bees
* **F**: Foragers
* **R**: Food Resources

## Key Features

### 1. Biological Realism
* **Seasonality (s(t)):** Implements a sigmoidal seasonal forcing function that drives egg-laying rates and foraging success, simulating a full year cycle (Spring bloom vs. Winter dormancy).
* **Social Feedback Loops:**
    * **Inhibition:** High populations of foragers and larvae release pheromones that slow down the maturation of hive bees into foragers.
    * **Acceleration:** Resource scarcity or high mortality triggers a "precocious foraging" response, accelerating the transition rate ($T_H$).
* **Cannibalism:** Under extreme starvation (low $R$), the colony recycles larvae into food resources to sustain the adults.
* **Dynamic Mortality:** Death rates for larvae, hive bees, and foragers dynamically adjust based on food availability and seasonal stress (e.g., winter survival vs. summer burnout).

### 2. Numerical Solvers
The project includes three numerical integration methods to solve the differential equations:
* **Euler Method:** Simple first-order method.
* **Improved Euler (Heun's) Method:** Predictor-corrector method.
* **Fourth Order Runge-Kutta Method (RK4):** High-precision fourth-order method.

### 3. Visualization
Built-in plotting tools using `matplotlib` to visualize:
* Population curves for all castes.
* Resource levels on a secondary axis.
* Real-time fluctuations in the **Hive-to-Forager Transition Rate ($T_H$)**.

## Installation

### Prerequisites
You need Python 3.x and the following scientific libraries:

```bash
pip install numpy matplotlib
```

---

## Usage

Save `model.py`) and run it. The simulation loop is decoupled from the model logic, allowing you to swap solvers.

```python
import numpy as np
import matplotlib.pyplot as plt
from model import DyBePoModel, rk4

# 1. Define Initial State (aka Population)
initial_pops = {
    'E': 500, 'L': 800, 'P': 0, 'D': 0, 
    'H': 8000, 'F': 500, 'R': 8000
}

# 2. Initialize Model
hive = DyBePoModel(initial_pops)

# 3. Run Simulation Loop
days = 1000 # Total days simulates
dt = 0.5 # Time step (half-day)
history = []

t = 0 # Start from day 0 (January 1st), can be changed depending on needs
while t < days:
    # Calculate auxiliary variables (like Transition Rate) for analysis
    current_Th = hive._transition_hive_to_forager(
        hive.state['L'], hive.state['H'], hive.state['F'], 
        hive.state['R'], hive.state['D']
    )
    
    # Store history
    record = hive.state.copy()
    record['time'] = t
    record['T_H'] = current_Th
    history.append(record)
    
    # Calculate Next Step using RK4 Solver
    next_state = rk4(hive.state, t, dt, hive.get_derivatives)
    
    # Update State
    hive.state = next_state
    t += dt
```

---

## Model Documentation

### Differential Equations
The model solves the following system (derived from Section III of the paper):

1.  **Eggs ($\frac{dE}{dt}$):** Laid by queen (seasonal), removed by hatching or death.
2.  **Larvae ($\frac{dL}{dt}$):** Hatched from eggs, removed by pupation or dynamic mortality (starvation/cannibalism).
3.  **Pupae ($\frac{dP}{dt}$):** Metamorphosis stage.
4.  **Drones ($\frac{dD}{dt}$):** Male bees, ejected during winter.
5.  **Hive Bees ($\frac{dH}{dt}$):** Young adults, whose population depends on pupae emergence and transition to foraging.
6.  **Foragers ($\frac{dF}{dt}$):** Older adults, whose population depends on recruitment from hive bees ($T_H$) and environmental mortality.
7.  **Resources ($\frac{dR}{dt}$):** Net change = (Foraging Gain) - (Consumption) + (Cannibalism Gain).

### Key Functions
* `_seasonality(t)`: Returns a value between 0 and 1 representing seasonal productivity.
* `_cannibalism_factor(R)`: Returns $\phi$, the efficiency of converting larvae to food.
* `_transition_hive_to_forager(...)`: Calculates $T_H$, the rate at which hive bees become foragers.

---

## Citation
This code is an implementation of the model described in:

> **DyBePo: Dynamic Beehive Population Modeling**
> Joey Yizhi Li, Sheng (Bob) Dai, Jiakai (Eric) Wei, Paul Dong
> *University of Toronto*
