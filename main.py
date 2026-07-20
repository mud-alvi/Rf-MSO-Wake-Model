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
from wake_model import wake_deficit, combined_wind_speed
from layouts import grid_layout, staggered_layout

np.random.seed(42)  # reproducibility

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "era5_wind_speeds2.csv")
SITE_LAT = 35.25    # nearest ERA5 grid point to Amarillo, TX (35.22N target)
SITE_LON = -101.75  # nearest ERA5 grid point to Amarillo, TX (-101.82W target)
#DATA_YEAR = 2023     # representative single year pulled from the multi-year file
START_YEAR = 2020
END_YEAR = 2025

# ---------------------------------------------------------------------
# STEP 1: Wind data (REAL ERA5 DATA)
# ---------------------------------------------------------------------
def load_real_wind_data(csv_path=CSV_PATH, lat=SITE_LAT, lon=SITE_LON, start_year=START_YEAR, end_year=END_YEAR):
    """
    Loads real ERA5 wind data for the Amarillo site.

    The raw file is a multi-location, multi-year grid (every 0.25 degrees
    lat/lon, 2015-2025), so this filters down to just the single grid
    point nearest the project's target coordinates and a range of
     years.

    Returns:
        speeds     : wind speed at 100m (m/s), array
        directions : wind direction in degrees (meteorological "from"
                     convention, 0=N, 90=E, 180=S, 270=W), derived from
                     the u100/v100 velocity components
        n_samples_per_year : how many readings were available for that
                     year (used later to correctly scale AEP, since this
                     dataset only has 4 readings/day -- 14:00-17:00 UTC --
                     not all 24 hours)
    """
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"])

    site = df[(df["latitude"] == lat) & (df["longitude"] == lon)].copy()
    site = site[(site["time"].dt.year.between(start_year, end_year))]

    if len(site) == 0:
        raise ValueError(
            f"No data found for lat={lat}, lon={lon}, years={start_year}-{end_year}. "
            f"Check that these match an actual grid point/year in the CSV."
        )

    speeds = site["wind_speed_ms"].values

    # ERA5 gives u/v velocity components, not direction directly.
    # Meteorological convention (direction wind is blowing FROM):
    u = site["u100"].values
    v = site["v100"].values
    directions = (np.degrees(np.arctan2(-u, -v))) % 360

    return speeds, directions, len(site)


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

        theta = np.radians(direction)
        # Rotate turbine coordinates so "x" always points downwind
        rot = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])
        rotated = positions @ rot.T
        rotated_positions = list(map(tuple, rotated))

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
