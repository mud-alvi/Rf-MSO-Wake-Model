"""Run the ERA5-driven 5x5 grid-versus-staggered wake experiment."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from layouts import grid_layout, staggered_layout
from turbine import vestas
from wake_model import combined_wind_speed, wake_deficit


np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "era5_wind_speeds2.csv")
SITE_LAT = 35.25
SITE_LON = -101.75
START_YEAR = 2020
END_YEAR = 2025


def load_real_wind_data(
    csv_path=CSV_PATH,
    lat=SITE_LAT,
    lon=SITE_LON,
    start_year=START_YEAR,
    end_year=END_YEAR,
):
    """Load Amarillo wind speeds and meteorological wind-from directions."""
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"])

    site = df[(df["latitude"] == lat) & (df["longitude"] == lon)].copy()
    site = site[site["time"].dt.year.between(start_year, end_year)]

    if len(site) == 0:
        raise ValueError(
            f"No data found for lat={lat}, lon={lon}, years={start_year}-{end_year}. "
            "Check that these match an actual grid point/year in the CSV."
        )

    speeds = site["wind_speed_ms"].to_numpy()
    u = site["u100"].to_numpy()
    v = site["v100"].to_numpy()
    directions = np.degrees(np.arctan2(-u, -v)) % 360
    return speeds, directions, len(site)


def calculate_prevailing_wind_direction(wind_directions, n_sectors=16):
    """Return the center of the most frequent meteorological wind sector.

    A dominant-sector calculation is used instead of a whole-dataset circular
    mean because opposing wind regimes can otherwise cancel each other out.
    The direction is refined by taking a circular mean inside the winning
    sector.
    """
    directions = np.asarray(wind_directions, dtype=float)
    directions = directions[np.isfinite(directions)] % 360
    if directions.size == 0:
        raise ValueError("Cannot calculate prevailing direction from empty data.")
    if n_sectors < 1:
        raise ValueError("n_sectors must be at least 1.")

    sector_width = 360.0 / n_sectors
    sector_ids = (
        np.floor((directions + sector_width / 2) / sector_width).astype(int)
        % n_sectors
    )
    dominant_sector = np.bincount(sector_ids, minlength=n_sectors).argmax()
    dominant_directions = directions[sector_ids == dominant_sector]

    mean_vector = np.mean(np.exp(1j * np.radians(dominant_directions)))
    return float(np.degrees(np.angle(mean_vector)) % 360)


def _to_wind_aligned_coordinates(positions, wind_from_direction):
    """Project geographic (east, north) coordinates onto down/crosswind axes."""
    positions = np.asarray(positions, dtype=float)
    direction = np.radians(float(wind_from_direction) % 360)

    # Meteorological wind direction states where wind comes FROM.
    downwind = np.array([-np.sin(direction), -np.cos(direction)])
    crosswind = np.array([np.cos(direction), -np.sin(direction)])
    return np.column_stack((positions @ downwind, positions @ crosswind))


def calculate_aep(positions, wind_speeds, wind_directions, hours_per_year=8760):
    """Calculate wake-inclusive farm AEP and average wake loss."""
    positions = np.asarray(positions, dtype=float)
    total_energy_kwh = 0.0
    n_valid_samples = 0
    cumulative_wake_loss = []

    for speed, direction in zip(wind_speeds, wind_directions):
        if speed < 1.0:
            continue

        rotated_positions = list(
            map(tuple, _to_wind_aligned_coordinates(positions, direction))
        )

        hour_power_kw = 0.0
        hour_freestream_power_kw = 0.0
        for x, y in rotated_positions:
            effective_speed = combined_wind_speed(
                rotated_positions, x, y, speed
            )
            hour_power_kw += vestas.power_output(effective_speed)
            hour_freestream_power_kw += vestas.power_output(speed)

        total_energy_kwh += hour_power_kw
        n_valid_samples += 1
        if hour_freestream_power_kw > 0:
            cumulative_wake_loss.append(
                1 - hour_power_kw / hour_freestream_power_kw
            )

    if n_valid_samples == 0:
        raise ValueError("No valid wind samples were available for AEP calculation.")

    avg_power_kw = total_energy_kwh / n_valid_samples
    aep_mwh = avg_power_kw * hours_per_year / 1000.0
    avg_wake_loss_pct = (
        np.mean(cumulative_wake_loss) * 100 if cumulative_wake_loss else 0.0
    )
    return aep_mwh, avg_wake_loss_pct


def run_experiment():
    print(
        f"Loading real ERA5 wind data for Amarillo, TX "
        f"({SITE_LAT}N, {SITE_LON}W), years {START_YEAR}-{END_YEAR}..."
    )
    speeds, directions, n_samples = load_real_wind_data()
    print(f"Loaded {n_samples} real wind samples.")

    prevailing_direction = calculate_prevailing_wind_direction(directions)
    print(
        f"Prevailing wind direction: {prevailing_direction:.1f} degrees "
        "(meteorological direction wind comes from)."
    )

    print("Building layouts aligned with the prevailing wind...")
    grid_positions = grid_layout(prevailing_direction)
    staggered_positions = staggered_layout(prevailing_direction)

    print("Running grid layout AEP calculation...")
    grid_aep, grid_wake_loss = calculate_aep(
        grid_positions, speeds, directions
    )

    print("Running staggered layout AEP calculation...")
    stag_aep, stag_wake_loss = calculate_aep(
        staggered_positions, speeds, directions
    )

    improvement_pct = (stag_aep - grid_aep) / grid_aep * 100

    print("\n===== RESULTS =====")
    print(
        f"Grid layout:      AEP = {grid_aep:,.1f} MWh | "
        f"avg wake loss = {grid_wake_loss:.1f}%"
    )
    print(
        f"Staggered layout: AEP = {stag_aep:,.1f} MWh | "
        f"avg wake loss = {stag_wake_loss:.1f}%"
    )
    print(
        f"Net AEP improvement (staggered vs grid): {improvement_pct:+.2f}%"
    )

    return {
        "speeds": speeds,
        "directions": directions,
        "prevailing_direction": prevailing_direction,
        "grid_positions": grid_positions,
        "staggered_positions": staggered_positions,
        "grid_aep": grid_aep,
        "stag_aep": stag_aep,
        "grid_wake_loss": grid_wake_loss,
        "stag_wake_loss": stag_wake_loss,
        "improvement_pct": improvement_pct,
    }


def make_wind_rose(speeds, directions, outpath):
    try:
        from windrose import WindroseAxes

        ax = WindroseAxes.from_ax()
        ax.bar(
            directions,
            speeds,
            normed=True,
            opening=0.8,
            edgecolor="white",
        )
        ax.set_legend()
        plt.title("Wind Rose - Amarillo, TX (ERA5 reanalysis data, 2020-2025)")
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close()
    except ImportError:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="polar")
        ax.hist(np.radians(directions), bins=16, weights=speeds)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        plt.title("Wind Rose - Amarillo, TX")
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close()


def make_wake_contour(outpath):
    diameter = vestas.rotor_diameter
    x = np.linspace(1, 15 * diameter, 200)
    y = np.linspace(-3 * diameter, 3 * diameter, 200)
    X, Y = np.meshgrid(x, y)
    deficit = wake_deficit(X, Y, U_inf=8.0)

    plt.figure(figsize=(9, 4))
    contour = plt.contourf(
        X / diameter, Y / diameter, deficit, levels=30, cmap="viridis_r"
    )
    plt.colorbar(contour, label="Velocity deficit (fraction)")
    plt.xlabel("Downstream distance (rotor diameters)")
    plt.ylabel("Crosswind distance (rotor diameters)")
    plt.title("Gaussian Wake Deficit Behind a Single Turbine")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def make_layout_comparison(
    grid_positions,
    staggered_positions,
    prevailing_direction,
    outpath,
):
    diameter = vestas.rotor_diameter
    grid = np.asarray(grid_positions) / diameter
    staggered = np.asarray(staggered_positions) / diameter

    direction = np.radians(prevailing_direction)
    downwind = np.array([-np.sin(direction), -np.cos(direction)])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    datasets = (
        (grid, "Grid Layout (5D x 5D)", "steelblue"),
        (staggered, "Staggered Layout", "darkorange"),
    )
    for ax, (positions, title, color) in zip(axes, datasets):
        ax.scatter(positions[:, 0], positions[:, 1], s=80, c=color)
        center = positions.mean(axis=0)
        arrow_end = center + downwind * 3
        ax.annotate(
            "Wind flow",
            xy=arrow_end,
            xytext=center,
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 2},
            ha="right",
            va="top",
            fontsize=9,
        )
        ax.set_title(title)
        ax.set_xlabel("Easting (rotor diameters)")
        ax.set_aspect("equal", adjustable="box")

    axes[0].set_ylabel("Northing (rotor diameters)")
    fig.suptitle(
        "Turbine Layouts Aligned to Prevailing Wind "
        f"({prevailing_direction:.1f} degrees from)"
    )
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def make_power_bar_chart(results, outpath):
    labels = ["Grid Layout", "Staggered Layout"]
    values = [results["grid_aep"], results["stag_aep"]]

    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, values, color=["steelblue", "darkorange"])
    plt.ylabel("Annual Energy Production (MWh)")
    plt.title(
        f"Layout Comparison: {results['improvement_pct']:+.2f}% AEP Change"
    )
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
        )
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    results = run_experiment()

    print("\nGenerating graphs...")
    make_wind_rose(
        results["speeds"], results["directions"], "graph1_wind_rose.png"
    )
    make_wake_contour("graph2_wake_contour.png")
    make_layout_comparison(
        results["grid_positions"],
        results["staggered_positions"],
        results["prevailing_direction"],
        "graph3_layout_comparison.png",
    )
    make_power_bar_chart(results, "graph4_power_comparison.png")
    print("Done. Graphs saved as graph1-4_*.png in the current directory.")
