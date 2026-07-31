"""Gaussian wake, yaw-steering, turbulence and fatigue-load helpers."""

import numpy as np

from turbine import vestas

WAKE_GROWTH_RATE = 0.075
AMBIENT_TI = 0.077


def wake_deficit(
    x,
    y,
    U_inf,
    D=None,
    Ct=None,
    k=WAKE_GROWTH_RATE,
    yaw_deg=0.0,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    D = vestas.rotor_diameter if D is None else float(D)
    Ct = vestas.ct_at(U_inf) if Ct is None else Ct
    Ct = float(np.clip(Ct, 0.0, 0.999))
    yaw = np.radians(np.clip(yaw_deg, -25.0, 25.0))
    ct_yaw = float(np.clip(Ct * np.cos(yaw) ** 2, 0.0, 0.999))

    deficit = np.zeros_like(x)
    valid = x > 0
    if not np.any(valid):
        return deficit

    root = np.sqrt(1.0 - ct_yaw)
    beta = 0.5 * (1.0 + root) / root
    sigma = k * x[valid] + 0.2 * np.sqrt(beta) * D
    theta = 0.3 * np.tan(yaw) * (1.0 - root) / (1.0 + root)
    lateral_position = y[valid] - theta * x[valid]
    term = np.clip(1.0 - ct_yaw / (8.0 * (sigma / D) ** 2), 0.0, None)
    deficit[valid] = (1.0 - np.sqrt(term)) * np.exp(
        -0.5 * (lateral_position / sigma) ** 2
    )
    return deficit


def wake_added_turbulence_intensity(x, Ct, I0=AMBIENT_TI, D=None):
    x = np.asarray(x, dtype=float)
    D = vestas.rotor_diameter if D is None else float(D)
    Ct = float(np.clip(Ct, 0.0, 0.999))
    a = (1.0 - np.sqrt(1.0 - Ct)) / 2.0
    wake_ti = np.zeros_like(x)
    valid = x > 0
    wake_ti[valid] = (
        0.73
        * a ** 0.8325
        * float(I0) ** -0.0325
        * (x[valid] / D) ** -0.32
    )
    return wake_ti


def _yaw_array(turbine_positions, yaw_angles):
    if yaw_angles is None:
        return np.zeros(len(turbine_positions))
    yaw_angles = np.asarray(yaw_angles, dtype=float)
    if yaw_angles.shape != (len(turbine_positions),):
        raise ValueError("yaw_angles must contain one angle per turbine.")
    return yaw_angles


def combined_wind_speed(
    turbine_positions,
    target_x,
    target_y,
    U_inf,
    yaw_angles=None,
):
    yaw_angles = _yaw_array(turbine_positions, yaw_angles)
    total_deficit_sq = 0.0
    for source_index, (tx, ty) in enumerate(turbine_positions):
        dx, dy = target_x - tx, target_y - ty
        if dx <= 0:
            continue
        deficit = wake_deficit(
            np.array([dx]),
            np.array([dy]),
            U_inf,
            yaw_deg=yaw_angles[source_index],
        )[0]
        total_deficit_sq += deficit ** 2
    return float(U_inf) * (1.0 - np.clip(np.sqrt(total_deficit_sq), 0.0, 1.0))


def effective_turbulence_intensity(
    turbine_positions,
    target_x,
    target_y,
    U_inf,
    ambient_ti=AMBIENT_TI,
    D=None,
    yaw_angles=None,
):
    yaw_angles = _yaw_array(turbine_positions, yaw_angles)
    total_added_sq = 0.0
    for source_index, (tx, ty) in enumerate(turbine_positions):
        dx, dy = target_x - tx, target_y - ty
        if dx <= 0:
            continue
        yaw = yaw_angles[source_index]
        Ct = float(vestas.ct_at(U_inf))
        root = np.sqrt(1.0 - Ct * np.cos(np.radians(yaw)) ** 2)
        centre_y = (
            0.3
            * np.tan(np.radians(yaw))
            * (1.0 - root)
            / (1.0 + root)
            * dx
        )
        centre = wake_deficit(
            np.array([dx]), np.array([centre_y]), U_inf, D=D, Ct=Ct, yaw_deg=yaw
        )[0]
        local = wake_deficit(
            np.array([dx]), np.array([dy]), U_inf, D=D, Ct=Ct, yaw_deg=yaw
        )[0]
        if centre <= 0:
            continue
        overlap = np.clip(local / centre, 0.0, 1.0)
        ct_yaw = Ct * np.cos(np.radians(yaw)) ** 2
        added = wake_added_turbulence_intensity(
            np.array([dx]), ct_yaw, ambient_ti, D
        )[0]
        total_added_sq += (added * overlap) ** 2
    return float(np.sqrt(float(ambient_ti) ** 2 + total_added_sq))


def tower_base_bending_moment(thrust_force, hub_height=None):
    hub_height = vestas.hub_height if hub_height is None else hub_height
    return np.asarray(thrust_force, dtype=float) * float(hub_height)


def tower_base_load_proxy(thrust_force, ti_eff, hub_height=None):
    return tower_base_bending_moment(thrust_force, hub_height) * np.asarray(
        ti_eff, dtype=float
    )


def tower_base_load_proxy_at_point(
    turbine_positions,
    target_x,
    target_y,
    U_inf,
    ambient_ti=AMBIENT_TI,
    rho=1.225,
    D=None,
    hub_height=None,
    yaw_angles=None,
    target_yaw_deg=0.0,
):
    effective_speed = combined_wind_speed(
        turbine_positions, target_x, target_y, U_inf, yaw_angles
    )
    ti_eff = effective_turbulence_intensity(
        turbine_positions,
        target_x,
        target_y,
        U_inf,
        ambient_ti,
        D,
        yaw_angles,
    )
    thrust = vestas.thrust_force(effective_speed, air_density=rho)
    thrust *= np.cos(np.radians(target_yaw_deg)) ** 2
    return tower_base_load_proxy(thrust, ti_eff, hub_height)
