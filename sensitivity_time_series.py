import numpy as np
import matplotlib.pyplot as plt
from model import DyBePoModel, rk4

def run_simulation_full_history(initial_state, params=None, dt=0.1, days=1000):
    """
    Run a simulation and return the full history.
    
    Args:
        initial_state (dict): Starting populations
        params (dict): Parameters to override
        dt (float): Time step for simulation
        days (int): Number of days to simulate
        
    Returns:
        list: History of states with time
    """
    # Create model instance
    hive = DyBePoModel(initial_state, params)
    
    # Run simulation
    t = 0
    history = []
    
    while t < days:
        history.append({**hive.state, 'time': t})
        next_state = rk4(hive.state, t, dt, hive.get_derivatives)
        hive.state = next_state
        t += dt
    
    return history

def plot_population_sensitivity(param_name, param_values, initial_state, 
                                population_type='total_adults', 
                                dt=0.1, days=1000, 
                                time_range=None, figsize=(12, 8)):
    """
    Plot population over time for different parameter values.
    
    Args:
        param_name (str): Name of parameter to vary (e.g., 'mu', 'healthy_ratio', 'b')
        param_values (list): List of parameter values to test
        initial_state (dict): Starting populations
        population_type (str): Type of population to plot
            - 'total_adults': H + F (hive bees + foragers)
            - 'total_pop': E + L + P + D + H + F (all populations)
            - 'H': Hive bees only
            - 'F': Foragers only
            - 'D': Drones only
            - 'L': Larvae only
            - 'R': Resources
        dt (float): Time step
        days (int): Number of days to simulate
        time_range (tuple): (start_day, end_day) to zoom in on specific time range
        figsize (tuple): Figure size
    """
    # Define population calculation function
    def get_population(history, pop_type):
        if pop_type == 'total_adults':
            return [step['H'] + step['F'] for step in history]
        elif pop_type == 'total_pop':
            return [step['E'] + step['L'] + step['P'] + step['D'] + step['H'] + step['F'] 
                   for step in history]
        elif pop_type in ['E', 'L', 'P', 'D', 'H', 'F', 'R']:
            return [step[pop_type] for step in history]
        else:
            raise ValueError(f"Unknown population_type: {pop_type}")
    
    # Get population label
    pop_labels = {
        'total_adults': 'Number of adult bees',
        'total_pop': 'Total population',
        'H': 'Number of hive bees',
        'F': 'Number of foragers',
        'D': 'Number of drones',
        'L': 'Number of larvae',
        'R': 'Resources (grams)'
    }
    ylabel = pop_labels.get(population_type, population_type)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors and markers for different lines
    colors = ['red', 'blue', 'black', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
    
    # Run simulations for each parameter value
    print(f"Analyzing {param_name}...")
    for i, param_val in enumerate(param_values):
        print(f"  [{i+1}/{len(param_values)}] {param_name} = {param_val:.4f}", end=' ... ')
        
        # Run simulation with this parameter value
        params = {param_name: param_val}
        history = run_simulation_full_history(initial_state, params, dt, days)
        
        # Extract time and population
        times = [step['time'] for step in history]
        populations = get_population(history, population_type)
        
        # Filter time range if specified
        if time_range:
            start_day, end_day = time_range
            filtered_times = []
            filtered_pops = []
            for t, p in zip(times, populations):
                if start_day <= t <= end_day:
                    filtered_times.append(t)
                    filtered_pops.append(p)
            times = filtered_times
            populations = filtered_pops
        
        # Plot this line
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        # Use markers every N points to avoid clutter
        marker_every = max(1, len(times) // 50)  # Show ~50 markers total
        ax.plot(times, populations, 
               color=color, marker=marker, markersize=6, 
               markevery=marker_every, linewidth=2,
               label=f'{param_name} = {param_val:.3f}')
        
        print("Done")
    
    # Format plot
    ax.set_xlabel('Days', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(f'Population Dynamics: {ylabel} vs {param_name}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    filename = f'sensitivity_{param_name}_{population_type}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {filename}\n")
    plt.show()

def main():
    """
    Main function to run sensitivity analysis on mu, healthy_ratio, and b.
    """
    # Initial state (same as in model.py)
    initial_pops = {
        'E': 900,
        'L': 1800,
        'P': 2000,
        'D': 100,
        'H': 1500,
        'F': 1200,
        'R': 5000
    }
    
    print("="*80)
    print("TIME SERIES SENSITIVITY ANALYSIS FOR BEE POPULATION MODEL")
    print("="*80)
    print()
    
    # 1. Analyze mu (drone ratio)
    print("\n" + "="*80)
    print("1. SENSITIVITY ANALYSIS: mu (Drone Ratio)")
    print("="*80)
    mu_values = [0.01, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
    plot_population_sensitivity('mu', mu_values, initial_pops, 
                               population_type='total_pop',
                               days=1000)
    plot_population_sensitivity('mu', mu_values, initial_pops, 
                               population_type='R',
                               days=1000)
    
    # 2. Analyze healthy_ratio
    print("\n" + "="*80)
    print("2. SENSITIVITY ANALYSIS: healthy_ratio")
    print("="*80)
    healthy_ratio_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    plot_population_sensitivity('healthy_ratio', healthy_ratio_values, initial_pops,
                               population_type='total_pop',
                               days=1000)
    plot_population_sensitivity('healthy_ratio', healthy_ratio_values, initial_pops,
                               population_type='R',
                               days=1000)
    
    # 3. Analyze b (cannibalism constant)
    print("\n" + "="*80)
    print("3. SENSITIVITY ANALYSIS: b (Cannibalism Constant)")
    print("="*80)
    b_values = [100, 250, 400, 500, 600, 750, 1000]
    plot_population_sensitivity('b', b_values, initial_pops,
                               population_type='total_pop',
                               days=1000)
    plot_population_sensitivity('b', b_values, initial_pops,
                               population_type='R',
                               days=1000)
    
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

