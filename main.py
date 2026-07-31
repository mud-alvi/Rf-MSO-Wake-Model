"""Weather loading and shared AEP/fatigue simulation for Amarillo layouts."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fatigue_model import AMBIENT_TI, FATIGUE_EXPONENT, fatigue_index
from layouts import grid_layout, staggered_layout
from turbine import vestas
from wake_model import (
    WAKE_GROWTH_RATE,
    combined_wind_speed,
    effective_turbulence_intensity,
    tower_base_load_proxy,
    wake_deficit,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WIND_CSV_PATH = os.path.join(SCRIPT_DIR, "era5_wind_speeds2.csv")
WEATHER_CSV_PATH = os.path.join(
    SCRIPT_DIR, "amarillo_temperature_pressure_2020_2025_hourly.csv"
)
TEMPERATURE_CSV_PATH = WEATHER_CSV_PATH
PRESSURE_CSV_PATH = WEATHER_CSV_PATH
SITE_LAT = 35.25
SITE_LON = -101.75
START_YEAR = 2020
END_YEAR = 2025

AIR_GAS_CONSTANT = 287.05
STANDARD_AIR_DENSITY = 1.225
MIN_AIR_DENSITY_KG_M3 = 0.2
MAX_AIR_DENSITY_KG_M3 = 2.0
INHG_TO_PA = 3386.389
REFERENCE_HEIGHT_M = 100.0
ROUGHNESS_LENGTH_M = 0.03
YAW_POWER_EXPONENT = 1.88


def _ensure_timestamp(df, column_name, allow_duplicates=False):
    if column_name not in df.columns:
        raise ValueError(f"Expected timestamp column '{column_name}'.")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df[column_name], errors="coerce", utc=True)
    if df["timestamp"].isna().any():
        rows = df.index[df["timestamp"].isna()].tolist()[:10]
        raise ValueError(f"Invalid timestamps in {column_name} at rows {rows}.")
    if not allow_duplicates and df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found after site filtering.")
    return df.sort_values("timestamp").reset_index(drop=True)


def _numeric_series(series):
    extracted = series.astype(str).str.extract(
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))", expand=False
    )
    return pd.to_numeric(extracted, errors="coerce")


def _to_kelvin(temperature_series, unit="F"):
    temperature = _numeric_series(temperature_series)
    if temperature.isna().all():
        raise ValueError("Temperature column contains no numeric values.")
    unit = unit.upper()
    if unit == "F":
        return (temperature - 32.0) * (5.0 / 9.0) + 273.15
    if unit == "C":
        return temperature + 273.15
    if unit == "K":
        return temperature
    raise ValueError("Temperature unit must be 'F', 'C', or 'K'.")


def _to_pascals(pressure_series, unit="inHg"):
    pressure = _numeric_series(pressure_series)
    if pressure.isna().all():
        raise ValueError("Pressure column contains no numeric values.")
    unit = unit.lower()
    if unit == "inhg":
        return pressure * INHG_TO_PA
    if unit == "hpa":
        return pressure * 100.0
    if unit == "pa":
        return pressure
    raise ValueError("Pressure unit must be 'inHg', 'hPa', or 'Pa'.")


def _calculate_air_density(pressure_pa, temperature_k):
    pressure = pd.to_numeric(pressure_pa, errors="coerce")
    temperature = pd.to_numeric(temperature_k, errors="coerce")
    density = pressure / (AIR_GAS_CONSTANT * temperature)
    invalid = density.notna() & (
        (density < MIN_AIR_DENSITY_KG_M3)
        | (density > MAX_AIR_DENSITY_KG_M3)
    )
    if invalid.any():
        raise ValueError(
            f"Impossible air density values: {density[invalid].head(10).tolist()}."
        )
    return density


def _load_era5_wind_data(csv_path, lat, lon, start_year, end_year):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Wind CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if {"latitude", "longitude"}.issubset(df.columns):
        latitudes = pd.to_numeric(df["latitude"], errors="coerce")
        longitudes = pd.to_numeric(df["longitude"], errors="coerce")
        df = df[np.isclose(latitudes, lat) & np.isclose(longitudes, lon)].copy()
    df = _ensure_timestamp(df, "time")
    df = df[df["timestamp"].dt.year.between(start_year, end_year)].copy()
    required = {"wind_speed_ms", "u100", "v100"}
    if not required.issubset(df.columns):
        raise ValueError(f"ERA5 wind CSV requires {sorted(required)}.")
    if df.empty:
        raise ValueError("No wind rows match the selected site and years.")
    df["wind_speed_m_s"] = pd.to_numeric(df["wind_speed_ms"], errors="coerce")
    df["wind_direction_deg"] = (
        np.degrees(
            np.arctan2(
                -pd.to_numeric(df["u100"], errors="coerce"),
                -pd.to_numeric(df["v100"], errors="coerce"),
            )
        )
        % 360.0
    )
    return df[["timestamp", "wind_speed_m_s", "wind_direction_deg"]]


def _load_noaa_weather_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Weather CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    time_column = next(
        (column for column in ("DATE", "date", "time") if column in df.columns),
        None,
    )
    if time_column is None:
        raise ValueError("NOAA weather CSV requires DATE, date, or time.")
    df = _ensure_timestamp(df, time_column, allow_duplicates=True)

    if "HourlyDryBulbTemperature_C" in df:
        temperature = _to_kelvin(df["HourlyDryBulbTemperature_C"], "C")
    elif "HourlyDryBulbTemperature" in df:
        temperature = _to_kelvin(df["HourlyDryBulbTemperature"], "F")
    elif "TAVG" in df:
        temperature = _to_kelvin(df["TAVG"], "F")
    elif {"TMAX", "TMIN"}.issubset(df.columns):
        temperature = _to_kelvin(
            (_numeric_series(df["TMAX"]) + _numeric_series(df["TMIN"])) / 2.0,
            "F",
        )
    else:
        raise ValueError(
            "NOAA file requires HourlyDryBulbTemperature_C, "
            "HourlyDryBulbTemperature, TAVG, or TMAX/TMIN."
        )

    if "HourlyStationPressure_hPa" in df:
        pressure = _to_pascals(df["HourlyStationPressure_hPa"], "hPa")
    elif "HourlyStationPressure" in df:
        pressure = _to_pascals(df["HourlyStationPressure"], "inHg")
    elif "HourlySeaLevelPressure" in df:
        pressure = _to_pascals(df["HourlySeaLevelPressure"], "inHg")
    else:
        raise ValueError(
            "NOAA file requires HourlyStationPressure_hPa, "
            "HourlyStationPressure, or HourlySeaLevelPressure."
        )

    daily = pd.DataFrame(
        {
            "date": df["timestamp"].dt.date,
            "temperature_K": temperature,
            "pressure_Pa": pressure,
        }
    )
    daily = daily.groupby("date", as_index=False).mean(numeric_only=True)

    # Fill short daily gaps from the surrounding measured values.
    weather_columns = ["temperature_K", "pressure_Pa"]
    daily[weather_columns] = daily[weather_columns].interpolate(
        method="linear",
        limit_direction="both",
    )

    daily["air_density_kg_m3"] = _calculate_air_density(
        daily["pressure_Pa"], daily["temperature_K"]
    )
    return daily


def load_synchronized_weather_data(
    wind_csv_path=WIND_CSV_PATH,
    weather_csv_path=WEATHER_CSV_PATH,
    lat=SITE_LAT,
    lon=SITE_LON,
    start_year=START_YEAR,
    end_year=END_YEAR,
    **legacy_paths,
):
    # Accept the former keyword names so peer scripts do not break.
    weather_csv_path = legacy_paths.get("temperature_csv_path", weather_csv_path)
    pressure_path = legacy_paths.get("pressure_csv_path", weather_csv_path)
    if pressure_path != weather_csv_path:
        raise ValueError("Use one combined NOAA LCD file for temperature and pressure.")

    wind = _load_era5_wind_data(
        wind_csv_path, lat, lon, start_year, end_year
    ).assign(date=lambda frame: frame["timestamp"].dt.date)
    combined = wind.merge(
        _load_noaa_weather_data(weather_csv_path),
        on="date",
        how="inner",
        validate="many_to_one",
    ).drop(columns="date")
    required = [
        "wind_speed_m_s",
        "wind_direction_deg",
        "temperature_K",
        "pressure_Pa",
        "air_density_kg_m3",
    ]
    if combined.empty:
        raise ValueError("No matching dates were found between ERA5 and NOAA data.")
    missing = combined[required].isna().sum()
    if missing.any():
        raise ValueError(f"Missing synchronized weather values:\n{missing}")
    return combined[["timestamp", *required]].reset_index(drop=True)


def load_real_weather_data(**kwargs):
    return load_synchronized_weather_data(**kwargs)


def load_real_wind_data(
    csv_path=WIND_CSV_PATH,
    lat=SITE_LAT,
    lon=SITE_LON,
    start_year=START_YEAR,
    end_year=END_YEAR,
):
    weather = _load_era5_wind_data(csv_path, lat, lon, start_year, end_year)
    return (
        weather["wind_speed_m_s"].to_numpy(),
        weather["wind_direction_deg"].to_numpy(),
        len(weather),
    )


def wind_speed_at_hub(
    reference_speed,
    reference_height=REFERENCE_HEIGHT_M,
    hub_height=vestas.hub_height,
    roughness_length=ROUGHNESS_LENGTH_M,
):
    if min(reference_height, hub_height, roughness_length) <= 0:
        raise ValueError("Wind-shear heights and roughness must be positive.")
    return np.asarray(reference_speed, dtype=float) * (
        np.log(hub_height / roughness_length)
        / np.log(reference_height / roughness_length)
    )


def build_wind_rose_cases(
    speeds,
    directions,
    densities=None,
    direction_bins=16,
    speed_bins=8,
):
    speeds = np.asarray(speeds, dtype=float)
    directions = np.asarray(directions, dtype=float) % 360.0
    densities = (
        np.full(len(speeds), STANDARD_AIR_DENSITY)
        if densities is None
        else np.asarray(densities, dtype=float)
    )
    if not (len(speeds) == len(directions) == len(densities)):
        raise ValueError("Wind speed, direction, and density lengths must match.")
    direction_edges = np.linspace(0.0, 360.0, direction_bins + 1)
    speed_edges = np.linspace(0.0, max(25.0, float(speeds.max())), speed_bins + 1)
    direction_index = np.clip(
        np.digitize(directions, direction_edges, right=False) - 1,
        0,
        direction_bins - 1,
    )
    speed_index = np.clip(
        np.digitize(speeds, speed_edges, right=False) - 1, 0, speed_bins - 1
    )
    cases = []
    for direction_bin in range(direction_bins):
        sector_direction = (
            direction_edges[direction_bin] + direction_edges[direction_bin + 1]
        ) / 2.0
        for speed_bin in range(speed_bins):
            selected = (direction_index == direction_bin) & (
                speed_index == speed_bin
            )
            if np.any(selected):
                cases.append(
                    (
                        float(speeds[selected].mean()),
                        float(sector_direction),
                        float(densities[selected].mean()),
                        float(selected.sum()),
                    )
                )
    return cases


def rotate_layout_to_wind_frame(positions, wind_direction_deg):
    theta = np.radians(wind_direction_deg)
    rotation = np.array(
        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    )
    return np.asarray(positions, dtype=float) @ rotation.T


def layout_wake_relations(
    positions,
    wind_direction_deg,
    Ct=None,
    D=None,
    k=WAKE_GROWTH_RATE,
):
    rotated = rotate_layout_to_wind_frame(positions, wind_direction_deg)
    return [
        {
            "index": target,
            "position": positions[target],
            "rotated_x": rotated[target, 0],
            "rotated_y": rotated[target, 1],
            "upstream_indices": [
                source
                for source in range(len(rotated))
                if source != target and rotated[target, 0] > rotated[source, 0]
            ],
            "downstream_indices": [
                source
                for source in range(len(rotated))
                if source != target and rotated[target, 0] <= rotated[source, 0]
            ],
        }
        for target in range(len(rotated))
    ]


def _sector_yaw(direction, yaw_schedule):
    if yaw_schedule is None:
        return 0.0
    schedule = np.asarray(yaw_schedule, dtype=float)
    sector = int((direction % 360.0) / (360.0 / len(schedule)))
    return float(schedule[min(sector, len(schedule) - 1)])


def _active_yaw_angles(rotated_positions, speed, sector_yaw):
    yaw_angles = np.zeros(len(rotated_positions))
    if np.isclose(sector_yaw, 0.0):
        return yaw_angles
    for source, (source_x, source_y) in enumerate(rotated_positions):
        for target_x, target_y in rotated_positions:
            dx, dy = target_x - source_x, target_y - source_y
            if dx > 0 and wake_deficit(
                np.array([dx]), np.array([dy]), speed
            )[0] > 1e-4:
                yaw_angles[source] = sector_yaw
                break
    return yaw_angles


def calculate_layout_performance(
    positions,
    wind_speeds,
    wind_directions,
    air_densities=None,
    sample_weights=None,
    yaw_schedule=None,
    hours_per_year=8760,
    ambient_ti=AMBIENT_TI,
    exponent=FATIGUE_EXPONENT,
    include_fatigue=True,
):
    positions = np.asarray(positions, dtype=float)
    speeds = wind_speed_at_hub(wind_speeds)
    directions = np.asarray(wind_directions, dtype=float)
    densities = (
        np.full(len(speeds), STANDARD_AIR_DENSITY)
        if air_densities is None
        else np.asarray(air_densities, dtype=float)
    )
    weights = (
        np.ones(len(speeds))
        if sample_weights is None
        else np.asarray(sample_weights, dtype=float)
    )
    if not (len(speeds) == len(directions) == len(densities) == len(weights)):
        raise ValueError("Weather arrays and sample weights must have equal length.")
    if len(speeds) == 0 or weights.sum() <= 0:
        raise ValueError("No weighted weather samples were supplied.")

    sample_power = np.zeros(len(speeds))
    sample_free_power = np.zeros(len(speeds))
    proxies = np.zeros((len(positions), len(speeds))) if include_fatigue else None

    for sample, (speed, direction, rho) in enumerate(
        zip(speeds, directions, densities)
    ):
        rotated = rotate_layout_to_wind_frame(positions, direction)
        yaw_angles = _active_yaw_angles(
            rotated, speed, _sector_yaw(direction, yaw_schedule)
        )
        density_factor = (rho / STANDARD_AIR_DENSITY) ** (1.0 / 3.0)
        for turbine, (x, y) in enumerate(rotated):
            effective_speed = combined_wind_speed(
                rotated, x, y, speed, yaw_angles
            )
            yaw_loss = np.cos(np.radians(yaw_angles[turbine])) ** YAW_POWER_EXPONENT
            sample_power[sample] += (
                vestas.power_output(effective_speed * density_factor) * yaw_loss
            )
            sample_free_power[sample] += vestas.power_output(speed * density_factor)
            if include_fatigue:
                ti_eff = effective_turbulence_intensity(
                    rotated,
                    x,
                    y,
                    speed,
                    ambient_ti=ambient_ti,
                    yaw_angles=yaw_angles,
                )
                thrust = vestas.thrust_force(effective_speed, air_density=rho)
                thrust *= np.cos(np.radians(yaw_angles[turbine])) ** 2
                proxies[turbine, sample] = tower_base_load_proxy(thrust, ti_eff)

    aep = np.average(sample_power, weights=weights) * hours_per_year / 1000.0
    operating = sample_free_power > 0
    wake_loss = (
        np.average(
            1.0 - sample_power[operating] / sample_free_power[operating],
            weights=weights[operating],
        )
        * 100.0
        if np.any(operating)
        else 0.0
    )
    result = {"aep": float(aep), "wake_loss": float(wake_loss)}
    if include_fatigue:
        per_turbine = np.array(
            [
                fatigue_index(proxies[index], weights=weights, exponent=exponent)
                for index in range(len(positions))
            ]
        )
        result.update(
            {
                "per_turbine_fatigue": per_turbine,
                "maximum_fatigue": float(per_turbine.max()),
                "mean_fatigue": float(per_turbine.mean()),
                "worst_turbine": int(per_turbine.argmax()),
            }
        )
    return result


def calculate_aep(
    positions,
    wind_speeds,
    wind_directions,
    air_densities=None,
    sample_weights=None,
    yaw_schedule=None,
    hours_per_year=8760,
):
    result = calculate_layout_performance(
        positions,
        wind_speeds,
        wind_directions,
        air_densities,
        sample_weights,
        yaw_schedule,
        hours_per_year,
        include_fatigue=False,
    )
    return result["aep"], result["wake_loss"]


def calculate_farm_fatigue(
    positions,
    wind_speeds,
    wind_directions,
    air_densities=None,
    ambient_ti=AMBIENT_TI,
    exponent=FATIGUE_EXPONENT,
    sample_weights=None,
    yaw_schedule=None,
):
    result = calculate_layout_performance(
        positions,
        wind_speeds,
        wind_directions,
        air_densities,
        sample_weights,
        yaw_schedule,
        ambient_ti=ambient_ti,
        exponent=exponent,
        include_fatigue=True,
    )
    return {
        key: result[key]
        for key in (
            "per_turbine_fatigue",
            "maximum_fatigue",
            "mean_fatigue",
            "worst_turbine",
        )
    }


def run_experiment():
    weather = load_real_weather_data()
    speeds = weather["wind_speed_m_s"].to_numpy()
    directions = weather["wind_direction_deg"].to_numpy()
    densities = weather["air_density_kg_m3"].to_numpy()
    results = {}
    for name, layout in (
        ("grid", grid_layout()),
        ("staggered", staggered_layout()),
    ):
        results[name] = calculate_layout_performance(
            layout, speeds, directions, densities
        )
    print(
        f"Grid AEP: {results['grid']['aep']:,.1f} MWh | "
        f"Staggered AEP: {results['staggered']['aep']:,.1f} MWh"
    )
    return results


if __name__ == "__main__":
    run_experiment()
