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
 
 
def wake_deficit(x, y, U_inf, D=None, Ct=None, k=WAKE_GROWTH_RATE):
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
 
    if D is None:
        D = vestas.rotor_diameter
    if Ct is None:
        Ct = vestas.ct_at(U_inf)
    Ct = float(np.clip(Ct, 0.0, 1.0))
 
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
 
 
AMBIENT_TI = 0.077
 
 
def wake_added_turbulence_intensity(x, Ct, I0=AMBIENT_TI, D=None):
    """Calculates the wake-added turbulence intensity from one turbine.
 
    This uses the Crespo–Hernández model:
      I+ = 0.73 * a^0.8325 * I0^-0.0325 * (x/D)^-0.32
    where a = (1 - sqrt(1 - Ct)) / 2.
    """
    x = np.asarray(x, dtype=float)
    if D is None:
        D = vestas.rotor_diameter
    Ct = float(np.clip(Ct, 0.0, 1.0))
    a = (1.0 - np.sqrt(1.0 - Ct)) / 2.0
    I0 = float(I0)
 
    wake_ti = np.zeros_like(x, dtype=float)
    valid = x > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        wake_ti[valid] = (
            0.73
            * a ** 0.8325
            * I0 ** (-0.0325)
            * (x[valid] / D) ** (-0.32)
        )
 
    return wake_ti
 
 
def effective_turbulence_intensity(
    turbine_positions,
    target_x,
    target_y,
    U_inf,
    ambient_ti=AMBIENT_TI,
    D=None,
):
    """Calculates effective turbulence intensity (TIeff) at a target point.
 
    Combines ambient turbulence with the sum-of-squares of wake-added
    contributions from upstream turbines.
    """
    total_i_plus_sq = 0.0
    for (tx, ty) in turbine_positions:
        dx = target_x - tx
        dy = target_y - ty
        if dx <= 0:
            continue
        Ct = vestas.ct_at(U_inf)
        i_plus = wake_added_turbulence_intensity(
            np.array([dx]), Ct, ambient_ti, D
        )[0]
        total_i_plus_sq += i_plus ** 2
 
    return np.sqrt(ambient_ti ** 2 + total_i_plus_sq)
 
 
def tower_base_bending_moment(thrust_force, hub_height=None):
    """Calculate the tower-base bending moment from thrust force and hub height."""
    if hub_height is None:
        hub_height = vestas.hub_height
    return np.asarray(thrust_force, dtype=float) * float(hub_height)
 
 
def tower_base_load_proxy(thrust_force, ti_eff, hub_height=None):
    """Calculate a turbulence-driven cyclic tower-base load proxy."""
    moment = tower_base_bending_moment(thrust_force, hub_height)
    return moment * np.asarray(ti_eff, dtype=float)
 
 
def tower_base_load_proxy_at_point(
    turbine_positions,
    target_x,
    target_y,
    U_inf,
    ambient_ti=AMBIENT_TI,
    rho=1.225,
    D=None,
    hub_height=None,
):
    """Calculate the tower-base load proxy for a single turbine in a layout."""
    Ueff = combined_wind_speed(turbine_positions, target_x, target_y, U_inf)
    ti_eff = effective_turbulence_intensity(
        turbine_positions,
        target_x,
        target_y,
        U_inf,
        ambient_ti=ambient_ti,
        D=D,
    )
    thrust = vestas.thrust_force(Ueff, air_density=rho)
    return tower_base_load_proxy(thrust, ti_eff, hub_height)
 
 
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