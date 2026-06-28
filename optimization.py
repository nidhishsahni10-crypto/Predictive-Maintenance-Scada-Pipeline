import numpy as np
from scipy.optimize import minimize

def cost_objective_function(operational_inputs):
    """
    Mathematical cost model for the plant environment.
    operational_inputs[0] = Inductor Exhaust Fan Speed (RPM)
    operational_inputs[1] = Coolant Supply Regulating Valve Opening (%)
    """
    fan_speed = operational_inputs[0]
    coolant_valve = operational_inputs[1]
    
    # Cost equation: Quadratic pricing curve representing operational energy drawing overhead
    total_energy_expenditure = (0.005 * (fan_speed ** 2)) + (1.4 * coolant_valve)
    return total_energy_expenditure

def quality_yield_floor_constraint(operational_inputs):
    """
    Ensures that process modifications do not degrade output parameters below a 95% floor.
    Calculates operational tolerances against product line thresholds.
    """
    fan_speed = operational_inputs[0]
    coolant_valve = operational_inputs[1]
    
    # Physics approximation engine: High fan speed drops thermal error, boosting yield metrics
    calculated_yield = 100.0 - (1650.0 / (fan_speed + 1.0)) - (0.45 * (coolant_valve - 5.0) ** 2)
    
    # Scipy expects constraints to be structured as: Actual Value - Threshold Value >= 0
    return calculated_yield - 95.0

if __name__ == "__main__":
    print("\n[Executing Non-Linear Energy Cost Optimization Loop]")
    
    # Baseline Strategy: Standard manual operational presets (High speeds, open valves)
    legacy_presets = np.array([480.0, 12.5])
    legacy_cost = cost_objective_function(legacy_presets)
    
    # Set operational thresholds and search boundaries
    operational_boundaries = ((120.0, 600.0), (2.0, 25.0))
    constraint_configuration = {'type': 'ineq', 'fun': quality_yield_floor_constraint}
    
    # Execute the optimization search using Sequential Least Squares Programming (SLSQP)
    optimization_execution = minimize(
        cost_objective_function,
        x0=[250.0, 6.0],
        method='SLSQP',
        bounds=operational_boundaries,
        constraints=constraint_configuration
    )
    
    optimized_cost = optimization_execution.fun
    calculated_savings = ((legacy_cost - optimized_cost) / legacy_cost) * 100
    
    print(f" -> Legacy Plant Operating Cost:   ${legacy_cost:.2f}")
    print(f" -> Algorithmic Optimized Cost:    ${optimized_cost:.2f}")
    print(f" -> Verified System Efficiency:   {calculated_savings:.2f}% Cost Reduction achieved")