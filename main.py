"""
Day 3-5 experiment: proves the Gaussian wake model + AEP pipeline works
on a small 5x5 array, comparing a standard grid layout vs a staggered
layout, per the feedback plan.

UPDATE: Now uses REAL ERA5 reanalysis wind data (era5_wind_speeds2.csv)
for the grid point nearest Amarillo, TX (35.25N, -101.75W, ~2.5 km from
the target 35.22N/101.82W coordinates). Source: ERA5 reanalysis, 100m
wind, hourly (afternoon-window samples: 14:00-17:00 UTC daily), 2015-2025.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from turbine import vestas 
from wake_model import (
    wake_deficit,
    combined_wind_speed,
    effective_turbulence_intensity,
    tower_base_load_proxy_at_point,
    WAKE_GROWTH_RATE,
)
from fatigue_model import AMBIENT_TI, FATIGUE_EXPONENT, fatigue_index
from layouts import grid_layout, staggered_layout

np.random.seed(42)  # reproducibility

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WIND_CSV_PATH = os.path.join(SCRIPT_DIR, "era5_wind_speeds2023_only.csv")
TEMPERATURE_CSV_PATH = os.path.join(SCRIPT_DIR, "4356691.csv")
PRESSURE_CSV_PATH = TEMPERATURE_CSV_PATH
SITE_LAT = 35.25    # nearest ERA5 grid point to Amarillo, TX (35.22N target)
SITE_LON = -101.75  # nearest ERA5 grid point to Amarillo, TX (-101.82W target)
START_YEAR = 2020
END_YEAR = 2025
AIR_GAS_CONSTANT = 287.05  # J/(kg·K)
MIN_AIR_DENSITY_KG_M3 = 0.2
MAX_AIR_DENSITY_KG_M3 = 2.0

# ---------------------------------------------------------------------
# STEP 1: Synchronized weather dataset
# ---------------------------------------------------------------------

def _ensure_timestamp(df, column_name, tz="UTC"):
    if column_name not in df.columns:
        raise ValueError(f"Expected timestamp column '{column_name}' in dataset.")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df[column_name], errors="coerce", utc=True)
    if df["timestamp"].isna().any():
        bad_rows = df.index[df["timestamp"].isna()].tolist()[:10]
        raise ValueError(
            f"Invalid timestamps found in {column_name} at rows: {bad_rows}"
        )
    if df["timestamp"].duplicated().any():
        duplicates = (
            df.loc[df["timestamp"].duplicated(), "timestamp"]
            .astype(str)
            .tolist()[:10]
        )
        raise ValueError(f"Duplicate timestamps found: {duplicates}")
    return df.sort_values("timestamp").reset_index(drop=True)


def _to_kelvin(temperature_series):
    temperature = pd.to_numeric(temperature_series, errors="coerce")
    if temperature.isna().all():
        raise ValueError("Temperature column contains no numeric values.")

    if temperature.between(-100, 80).all():
        return temperature + 273.15
    if temperature.between(150, 350).all():
        return temperature
    raise ValueError(
        "Temperature values are outside expected °C or K ranges. "
        "Provide a consistent temperature dataset."
    )


def _to_pascals(pressure_series):
    pressure = pd.to_numeric(pressure_series, errors="coerce")
    if pressure.isna().all():
        raise ValueError("Pressure column contains no numeric values.")

    if pressure.between(100, 2000).all():
        return pressure * 100.0
    if pressure.between(2000, 200000).all():
        return pressure
    raise ValueError(
        "Pressure values are outside expected hPa or Pa ranges. "
        "Provide a consistent pressure dataset."
    )


def _calculate_air_density(pressure_pa, temperature_k):
    pressure = pd.to_numeric(pressure_pa, errors="coerce")
    temperature = pd.to_numeric(temperature_k, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        density = pressure / (AIR_GAS_CONSTANT * temperature)

    if not density.isin([np.nan]).all():
        bad_values = density[(density.notna()) & ((density < MIN_AIR_DENSITY_KG_M3) | (density > MAX_AIR_DENSITY_KG_M3))]
        if len(bad_values):
            example_values = bad_values.iloc[:10].to_list()
            raise ValueError(
                "Impossible air density values detected: "
                f"{example_values}. Check temperature and pressure units/validity."
            )

    return density


def _load_era5_wind_data(csv_path, lat, lon, start_year, end_year):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Wind CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = _ensure_timestamp(df, "time")
    if "latitude" in df.columns and "longitude" in df.columns:
        site = df[(df["latitude"] == lat) & (df["longitude"] == lon)].copy()
    else:
        site = df.copy()

    site = site[site["timestamp"].dt.year.between(start_year, end_year)]
    if site.empty:
        raise ValueError(
            f"No wind data found for lat={lat}, lon={lon}, years={start_year}-{end_year}."
        )

    if "wind_speed_ms" not in site.columns or "u100" not in site.columns or "v100" not in site.columns:
        raise ValueError("ERA5 wind CSV must contain wind_speed_ms, u100, and v100 columns.")

    site = site.copy()
    site["wind_speed_m_s"] = pd.to_numeric(site["wind_speed_ms"], errors="coerce")
    site["wind_direction_deg"] = (
        np.degrees(np.arctan2(-site["u100"].astype(float), -site["v100"].astype(float)))
        % 360
    )
    site = site[["timestamp", "wind_speed_m_s", "wind_direction_deg"]]
    return site


def rotate_layout_to_wind_frame(positions, wind_direction_deg):
    """Rotate a turbine layout into the wind-aligned coordinate system."""
    theta = np.radians(wind_direction_deg)
    rot = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)],
    ])
    rotated = np.dot(np.array(positions), rot.T)
    return [tuple(coord) for coord in rotated]


def layout_wake_relations(positions, wind_direction_deg, Ct=None, D=None, k=WAKE_GROWTH_RATE):
    """Return upstream/downstream relations and wake geometry in the wind-aligned frame."""
    if D is None:
        D = vestas.rotor_diameter
    if Ct is None:
        Ct = vestas.thrust_coefficient

    rotated = rotate_layout_to_wind_frame(positions, wind_direction_deg)
    if Ct < 0 or Ct >= 1:
        raise ValueError("Invalid thrust coefficient for wake geometry calculation.")

    beta = 0.5 * (1 + np.sqrt(1 - Ct)) / np.sqrt(1 - Ct)
    epsilon = 0.2 * np.sqrt(beta)
    info = []

    for target_idx, (x_t, y_t) in enumerate(rotated):
        upstream = []
        downstream = []
        for src_idx, (x_s, y_s) in enumerate(rotated):
            if src_idx == target_idx:
                continue
            dx = x_t - x_s
            dy = y_t - y_s
            if dx <= 0:
                upstream.append(src_idx)
            else:
                downstream.append(src_idx)

        info.append(
            {
                "index": target_idx,
                "position": positions[target_idx],
                "rotated_x": x_t,
                "rotated_y": y_t,
                "upstream_indices": sorted(upstream),
                "downstream_indices": sorted(downstream),
            }
        )

    return info


def _load_temperature_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Temperature CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    time_column = "DATE" if "DATE" in df.columns else ("date" if "date" in df.columns else None)
    if time_column is None:
        raise ValueError("Temperature CSV must contain a timestamp column named 'DATE' or 'date'.")

    df = _ensure_timestamp(df, time_column)
    if "TAVG" in df.columns:
        temperature = pd.to_numeric(df["TAVG"], errors="coerce")
    elif "TMAX" in df.columns and "TMIN" in df.columns:
        temperature = (
            pd.to_numeric(df["TMAX"], errors="coerce")
            + pd.to_numeric(df["TMIN"], errors="coerce")
        ) / 2.0
    else:
        raise ValueError("Temperature CSV must contain TAVG or both TMAX and TMIN columns.")

    temperature_kelvin = _to_kelvin(temperature)
    df = df.assign(temperature_K=temperature_kelvin)
    df["date"] = df["timestamp"].dt.date
    return df[["timestamp", "date", "temperature_K"]]


def _load_pressure_data(csv_path, pressure_column=None):
    if csv_path is None:
        return None
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Pressure CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    time_column = "time" if "time" in df.columns else ("DATE" if "DATE" in df.columns else ("date" if "date" in df.columns else None))
    if time_column is None:
        raise ValueError("Pressure CSV must contain a timestamp column named 'time', 'DATE' or 'date'.")

    df = _ensure_timestamp(df, time_column)
    if pressure_column is None:
        pressure_column = next(
            (col for col in df.columns if "press" in col.lower() or "pressure" in col.lower()),
            None,
        )
    if pressure_column is None:
        raise ValueError("Pressure CSV must contain a pressure column.")

    pressure_pa = _to_pascals(df[pressure_column])
    df = df.assign(pressure_Pa=pressure_pa)
    df["date"] = df["timestamp"].dt.date
    return df[["timestamp", "date", "pressure_Pa"]]


def load_synchronized_weather_data(
    wind_csv_path=WIND_CSV_PATH,
    temperature_csv_path=TEMPERATURE_CSV_PATH,
    pressure_csv_path=PRESSURE_CSV_PATH,
    lat=SITE_LAT,
    lon=SITE_LON,
    start_year=START_YEAR,
    end_year=END_YEAR,
):
    """Load and synchronize weather datasets on timestamp, returning a single validated dataset."""
    weather_df = _load_era5_wind_data(wind_csv_path, lat, lon, start_year, end_year)
    weather_df = weather_df.assign(date=weather_df["timestamp"].dt.date)

    temperature_df = None
    if temperature_csv_path is not None:
        temperature_df = _load_temperature_data(temperature_csv_path)
        weather_df = weather_df.merge(temperature_df, on="date", how="inner")

    pressure_df = None
    if pressure_csv_path is not None:
        pressure_df = _load_pressure_data(pressure_csv_path)
        weather_df = weather_df.merge(pressure_df, on="date", how="inner")

    if weather_df.empty:
        raise ValueError("No common timestamps found across the supplied weather datasets.")

    weather_df = weather_df.drop(columns=["date"])
    weather_df = weather_df[weather_df["timestamp"].dt.year.between(start_year, end_year)]

    if temperature_df is None:
        weather_df["temperature_K"] = np.nan
    if pressure_df is None:
        weather_df["pressure_Pa"] = np.nan

    if "temperature_K" in weather_df.columns and "pressure_Pa" in weather_df.columns:
        weather_df["air_density_kg_m3"] = _calculate_air_density(
            weather_df["pressure_Pa"],
            weather_df["temperature_K"],
        )
    else:
        weather_df["air_density_kg_m3"] = np.nan

    weather_df = weather_df[["timestamp", "wind_speed_m_s", "wind_direction_deg", "temperature_K", "pressure_Pa", "air_density_kg_m3"]]
    return weather_df


def load_real_weather_data(
    wind_csv_path=WIND_CSV_PATH,
    temperature_csv_path=TEMPERATURE_CSV_PATH,
    pressure_csv_path=PRESSURE_CSV_PATH,
    lat=SITE_LAT,
    lon=SITE_LON,
    start_year=START_YEAR,
    end_year=END_YEAR,
):
    return load_synchronized_weather_data(
        wind_csv_path=wind_csv_path,
        temperature_csv_path=temperature_csv_path,
        pressure_csv_path=pressure_csv_path,
        lat=lat,
        lon=lon,
        start_year=start_year,
        end_year=end_year,
    )


def load_real_wind_data(csv_path=WIND_CSV_PATH, lat=SITE_LAT, lon=SITE_LON, start_year=START_YEAR, end_year=END_YEAR):
    weather_df = _load_era5_wind_data(csv_path, lat, lon, start_year, end_year)
    return (
        weather_df["wind_speed_m_s"].values,
        weather_df["wind_direction_deg"].values,
        len(weather_df),
    )


# ---------------------------------------------------------------------
# STEP 2: AEP calculation for a given layout
# ---------------------------------------------------------------------
def calculate_aep(positions, wind_speeds, wind_directions, hours_per_year=8760):
    """
    Calculates total farm Annual Energy Production (AEP, in MWh) for a
    given turbine layout, accounting for wake losses turbine-by-turbine
    for every sample in the wind dataset.

    IMPORTANT SCALING NOTE: the real ERA5 data only has 4 samples/day
    (14:00-17:00 UTC, i.e. afternoon), not all 24 hours -- so AEP is
    estimated by computing the AVERAGE power per sample and scaling that
    up to a full year's worth of hours (8760). This is a standard
    approach for sample-based AEP estimation, but it does carry a real
    caveat worth stating in the paper: afternoon wind in the Panhandle
    may be systematically higher (or lower) than the true 24-hour
    average due to the diurnal wind cycle, so this AEP estimate should
    be flagged as based on a partial daily sampling window, not a true
    continuous hourly record.

    To keep runtime reasonable for a 5x5 array x many samples, we rotate
    the turbine layout into a "wind-aligned" frame for each sample rather
    than rotating the wake model itself -- mathematically equivalent,
    much faster.
    """
    positions = np.array(positions)
    total_energy_kwh = 0.0
    n_valid_samples = 0
    cumulative_wake_loss = []

    for speed, direction in zip(wind_speeds, wind_directions):
        if speed < 1.0:
            continue  # skip near-zero wind hours, negligible contribution

        rotated_positions = rotate_layout_to_wind_frame(positions, direction)

        hour_power_kw = 0.0
        hour_freestream_power_kw = 0.0
        for (x, y) in rotated_positions:
            eff_speed = combined_wind_speed(rotated_positions, x, y, speed)
            hour_power_kw += vestas.power_output(eff_speed)
            hour_freestream_power_kw += vestas.power_output(speed)

        total_energy_kwh += hour_power_kw  # 1 sample-hour * kW = kWh
        n_valid_samples += 1
        if hour_freestream_power_kw > 0:
            cumulative_wake_loss.append(
                1 - (hour_power_kw / hour_freestream_power_kw)
            )

    # Scale from "average power per sample" up to a full year of hours,
    # since the real dataset only samples 4 hours/day (see docstring note).
    avg_power_kw = total_energy_kwh / n_valid_samples
    aep_mwh = (avg_power_kw * hours_per_year) / 1000.0
    avg_wake_loss_pct = np.mean(cumulative_wake_loss) * 100
    return aep_mwh, avg_wake_loss_pct


def calculate_farm_fatigue(
    positions,
    wind_speeds,
    wind_directions,
    air_densities=None,
    ambient_ti=AMBIENT_TI,
    exponent=FATIGUE_EXPONENT,
):
    """Calculate farm fatigue metrics for a layout.

    Returns per-turbine fatigue values and farm summary metrics.
    If the dataset contains only four samples per day, this is a sampled
    exposure estimate rather than an absolute fatigue damage metric.
    """
    positions = np.array(positions)
    n_turbines = len(positions)

    if air_densities is None:
        air_densities = np.full(len(wind_speeds), 1.225, dtype=float)
    air_densities = np.asarray(air_densities, dtype=float)

    per_turbine_proxies = np.zeros((n_turbines, len(wind_speeds)), dtype=float)

    for sample_index, (speed, direction, rho) in enumerate(
        zip(wind_speeds, wind_directions, air_densities)
    ):
        rotated_positions = rotate_layout_to_wind_frame(positions, direction)
        for turbine_index, (x, y) in enumerate(rotated_positions):
            per_turbine_proxies[turbine_index, sample_index] = (
                tower_base_load_proxy_at_point(
                    rotated_positions,
                    x,
                    y,
                    speed,
                    ambient_ti=ambient_ti,
                    rho=rho,
                )
            )

    per_turbine_fatigue = np.array(
        [
            fatigue_index(per_turbine_proxies[t], exponent=exponent)
            for t in range(n_turbines)
        ]
    )
    maximum_fatigue = np.max(per_turbine_fatigue)
    mean_fatigue = np.mean(per_turbine_fatigue)
    worst_turbine = int(np.argmax(per_turbine_fatigue))

    return {
        "per_turbine_fatigue": per_turbine_fatigue,
        "maximum_fatigue": maximum_fatigue,
        "mean_fatigue": mean_fatigue,
        "worst_turbine": worst_turbine,
    }


# ---------------------------------------------------------------------
# STEP 3: Run the comparison
# ---------------------------------------------------------------------
def run_experiment():
    print(f"Loading real ERA5 wind data for Amarillo, TX ({SITE_LAT}N, {SITE_LON}W), years {START_YEAR}-{END_YEAR}...")
    speeds, directions, n_samples = load_real_wind_data()
    print(f"Loaded {n_samples} real wind samples (4 readings/day, 14:00-17:00 UTC).")

    print("Building layouts...")
    grid_positions = grid_layout()
    staggered_positions = staggered_layout()

    print("Running grid layout AEP calculation (this may take ~1 min)...")
    grid_aep, grid_wake_loss = calculate_aep(grid_positions, speeds, directions)

    print("Running staggered layout AEP calculation (this may take ~1 min)...")
    stag_aep, stag_wake_loss = calculate_aep(staggered_positions, speeds, directions)

    improvement_pct = ((stag_aep - grid_aep) / grid_aep) * 100

    print("\n===== RESULTS =====")
    print(f"Grid layout:      AEP = {grid_aep:,.1f} MWh | avg wake loss = {grid_wake_loss:.1f}%")
    print(f"Staggered layout: AEP = {stag_aep:,.1f} MWh | avg wake loss = {stag_wake_loss:.1f}%")
    print(f"Net AEP improvement (staggered vs grid): {improvement_pct:+.2f}%")

    return {
        "speeds": speeds,
        "directions": directions,
        "grid_positions": grid_positions,
        "staggered_positions": staggered_positions,
        "grid_aep": grid_aep,
        "stag_aep": stag_aep,
        "grid_wake_loss": grid_wake_loss,
        "stag_wake_loss": stag_wake_loss,
        "improvement_pct": improvement_pct,
    }


# ---------------------------------------------------------------------
# STEP 4: Generate the 4 required graphs
# ---------------------------------------------------------------------
def make_wind_rose(speeds, directions, outpath):
    try:
        from windrose import WindroseAxes
        ax = WindroseAxes.from_ax()
        ax.bar(directions, speeds, normed=True, opening=0.8, edgecolor="white")
        ax.set_legend()
        plt.title("Wind Rose - Amarillo, TX (ERA5 reanalysis data, 2020-2025)")
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close()
    except ImportError:
        # Fallback if the windrose package isn't installed: polar histogram
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="polar")
        theta = np.radians(directions)
        ax.hist(theta, bins=16, weights=speeds)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        plt.title("Wind Rose (fallback polar histogram) - install 'windrose' package for the standard version")
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close()


def make_wake_contour(outpath):
    D = vestas.rotor_diameter
    x = np.linspace(1, 15 * D, 200)
    y = np.linspace(-3 * D, 3 * D, 200)
    X, Y = np.meshgrid(x, y)
    deficit = wake_deficit(X, Y, U_inf=8.0)

    plt.figure(figsize=(9, 4))
    cp = plt.contourf(X / D, Y / D, deficit, levels=30, cmap="viridis_r")
    plt.colorbar(cp, label="Velocity deficit (fraction)")
    plt.xlabel("Downstream distance (rotor diameters)")
    plt.ylabel("Crosswind distance (rotor diameters)")
    plt.title("Gaussian Wake Deficit Behind a Single Turbine")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def make_layout_comparison(grid_positions, staggered_positions, outpath):
    D = vestas.rotor_diameter
    grid = np.array(grid_positions) / D
    stag = np.array(staggered_positions) / D

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    axes[0].scatter(grid[:, 0], grid[:, 1], s=80, c="steelblue")
    axes[0].set_title("Grid Layout (5D x 5D)")
    axes[0].set_xlabel("X (rotor diameters)")
    axes[0].set_ylabel("Y (rotor diameters)")

    axes[1].scatter(stag[:, 0], stag[:, 1], s=80, c="darkorange")
    axes[1].set_title("Staggered Layout")
    axes[1].set_xlabel("X (rotor diameters)")

    plt.suptitle("Turbine Layout Comparison")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def make_power_bar_chart(results, outpath):
    labels = ["Grid Layout", "Staggered Layout"]
    values = [results["grid_aep"], results["stag_aep"]]

    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, values, color=["steelblue", "darkorange"])
    plt.ylabel("Annual Energy Production (MWh)")
    plt.title(f"Layout Comparison: {results['improvement_pct']:+.2f}% AEP Change")

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, val, f"{val:,.0f}",
                  ha="center", va="bottom")

    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    results = run_experiment()

    print("\nGenerating graphs...")
    make_wind_rose(results["speeds"], results["directions"], "graph1_wind_rose.png")
    make_wake_contour("graph2_wake_contour.png")
    make_layout_comparison(results["grid_positions"], results["staggered_positions"], "graph3_layout_comparison.png")
    make_power_bar_chart(results, "graph4_power_comparison.png")
    print("Done. Graphs saved as graph1-4_*.png in the current directory.")
