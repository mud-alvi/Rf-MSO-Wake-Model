"""
Gaussian Wake Model (Bastankhah & Porte-Agel, 2014)

This is the wake model your methodology paper cites as the chosen model
for this project (over Jensen), because it captures wake deflection more
realistically than Jensen's simple top-hat wake shape.

Reference: Bastankhah, M., & Porte-Agel, F. (2014). A new analytical model
for wind-turbine wakes. Renewable Energy, 70, 116-123.
"""

import numpy as np
from turbine import vestas
 
# Wake expansion / growth rate. Typical literature value for onshore,
# moderate turbulence conditions is k = 0.075. This directly controls
# how quickly the wake widens and recovers with downstream distance --
# tune this later once you have real turbulence intensity data from
# Days 1-2, since higher turbulence intensity -> higher k -> faster
# wake recovery.
WAKE_GROWTH_RATE = 0.075
 
 
def wake_deficit(x, y, U_inf, D=vestas.rotor_diameter, Ct=vestas.thrust_coefficient, k=WAKE_GROWTH_RATE):
    """
    Calculates the fractional velocity deficit (0 to 1) at a point located
    x meters downstream and y meters laterally offset from a single
    turbine's wake centerline.
 
    x : downstream distance (m). Must be > 0 (this model is only valid
        downstream of the turbine).
    y : lateral (crosswind) offset from the wake centerline (m).
    U_inf : free-stream (undisturbed) wind speed (m/s).
 
    Returns: fractional deficit, so that the actual wind speed at that
    point is U_inf * (1 - deficit).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
 
    deficit = np.zeros_like(x, dtype=float)
    valid = x > 0
 
    beta = 0.5 * (1 + np.sqrt(1 - Ct)) / np.sqrt(1 - Ct)
    epsilon = 0.2 * np.sqrt(beta)
 
    sigma = k * x[valid] + epsilon * D  # wake width at each x
 
    # Core Bastankhah & Porte-Agel Gaussian deficit formula
    with np.errstate(invalid="ignore"):
        term_sqrt = 1 - (Ct / (8 * (sigma / D) ** 2))
        term_sqrt = np.clip(term_sqrt, 0, None)  # avoid negative sqrt
 
    deficit[valid] = (1 - np.sqrt(term_sqrt)) * np.exp(
        -0.5 * (y[valid] / sigma) ** 2
    )
 
    return deficit
 
 
def combined_wind_speed(turbine_positions, target_x, target_y, U_inf):
    """
    Calculates the effective wind speed at a target point, accounting for
    wakes from ALL upstream turbines using sum-of-squares superposition
    (the standard way to combine multiple overlapping wakes).
 
    turbine_positions : list of (x, y) tuples for every turbine in the farm
    target_x, target_y : location to evaluate (usually another turbine's position)
    U_inf : free-stream wind speed (m/s)
    """
    total_deficit_sq = 0.0
    for (tx, ty) in turbine_positions:
        dx = target_x - tx
        dy = target_y - ty
        if dx <= 0:
            continue  # turbines behind the target don't affect it
        d = wake_deficit(np.array([dx]), np.array([dy]), U_inf)[0]
        total_deficit_sq += d ** 2
 
    total_deficit = np.sqrt(total_deficit_sq)
    return U_inf * (1 - total_deficit)