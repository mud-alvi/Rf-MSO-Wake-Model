"""Weather loading and shared AEP/fatigue simulation for Amarillo layouts."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fatigue_model import AMBIENT_TI, FATIGUE_EXPONENT, fatigue_index
from layouts import grid_layout, staggered_layout
from lcoe_model import calculate_lcoe_details
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
COMPARISON_DOMAIN_D = 25
DIRECTION_BINS = 16

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


def centre_layout_in_domain(layout, domain_d=COMPARISON_DOMAIN_D):
    """Centre a layout inside the same 25D × 25D domain used by the GA."""
    layout = np.asarray(layout, dtype=float).copy()
    domain_size = domain_d * vestas.rotor_diameter

    layout -= layout.min(axis=0)
    extent = layout.max(axis=0)

    layout += np.array(
        [
            (domain_size - extent[0]) / 2.0,
            (domain_size - extent[1]) / 2.0,
        ]
    )

    return layout


def add_bar_labels(axis, bars, value_format):
    for bar in bars:
        value = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            value_format.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def make_metrics_comparison_graph(results):
    names = ["Grid", "Staggered"]
    colours = ["steelblue", "darkorange"]

    aep_values = [
        results["grid"]["aep"],
        results["staggered"]["aep"],
    ]

    lcoe_values = [
        results["grid"]["lcoe"],
        results["staggered"]["lcoe"],
    ]

    # Normalized because the raw fatigue proxy is around 10^29.
    grid_fatigue = results["grid"]["maximum_fatigue"]
    fatigue_ratios = [
        results["grid"]["maximum_fatigue"] / grid_fatigue,
        results["staggered"]["maximum_fatigue"] / grid_fatigue,
    ]

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))

    aep_bars = axes[0].bar(names, aep_values, color=colours)
    axes[0].set_title("Combined 25-Turbine AEP")
    axes[0].set_ylabel("AEP (MWh/year)")
    add_bar_labels(axes[0], aep_bars, "{:,.0f}")

    lcoe_bars = axes[1].bar(names, lcoe_values, color=colours)
    axes[1].set_title("Levelized Cost of Energy")
    axes[1].set_ylabel("LCOE ($/MWh)")
    add_bar_labels(axes[1], lcoe_bars, "${:.2f}")

    fatigue_bars = axes[2].bar(names, fatigue_ratios, color=colours)
    axes[2].axhline(1.0, color="black", linestyle="--", alpha=0.6)
    axes[2].set_title("Maximum Fatigue Proxy")
    axes[2].set_ylabel("Relative index (grid = 1.00)")
    add_bar_labels(axes[2], fatigue_bars, "{:.3f}")

    for axis in axes:
        axis.grid(axis="y", alpha=0.25)

    figure.suptitle(
        "Grid vs Staggered Performance — Amarillo 2020–2025"
    )
    figure.tight_layout()
    figure.savefig(
        os.path.join(SCRIPT_DIR, "grid_vs_staggered_metrics.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def make_wake_loss_graph(results):
    names = ["Grid", "Staggered"]
    wake_losses = [
        results["grid"]["wake_loss"],
        results["staggered"]["wake_loss"],
    ]

    figure, axis = plt.subplots(figsize=(8, 6))

    bars = axis.bar(
        names,
        wake_losses,
        color=["steelblue", "darkorange"],
    )

    axis.set_title("Grid vs Staggered Wake-Energy Loss")
    axis.set_ylabel("Wake loss (%)")
    axis.grid(axis="y", alpha=0.25)

    add_bar_labels(axis, bars, "{:.2f}%")

    figure.tight_layout()
    figure.savefig(
        os.path.join(
            SCRIPT_DIR,
            "grid_vs_staggered_wake_loss.png",
        ),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def make_wind_rose_graph(speeds, directions):
    # Convert ERA5 100 m speeds to the Vestas 95 m hub height.
    hub_speeds = wind_speed_at_hub(speeds)
    directions = np.asarray(directions, dtype=float) % 360.0

    direction_edges = np.linspace(
        0.0,
        360.0,
        DIRECTION_BINS + 1,
    )
    direction_centres = np.radians(
        (direction_edges[:-1] + direction_edges[1:]) / 2.0
    )
    bar_width = np.radians(360.0 / DIRECTION_BINS)

    speed_edges = np.array(
        [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 20.0, np.inf]
    )
    speed_labels = [
        "0–3 m/s",
        "3–6 m/s",
        "6–9 m/s",
        "9–12 m/s",
        "12–15 m/s",
        "15–20 m/s",
        "20+ m/s",
    ]

    colours = plt.cm.viridis(
        np.linspace(0.12, 0.92, len(speed_labels))
    )

    figure, axis = plt.subplots(
        figsize=(9, 8),
        subplot_kw={"projection": "polar"},
    )

    bottom = np.zeros(DIRECTION_BINS)

    for lower, upper, label, colour in zip(
        speed_edges[:-1],
        speed_edges[1:],
        speed_labels,
        colours,
    ):
        selected = (hub_speeds >= lower) & (hub_speeds < upper)

        counts, _ = np.histogram(
            directions[selected],
            bins=direction_edges,
        )

        frequency = counts / len(directions) * 100.0

        axis.bar(
            direction_centres,
            frequency,
            width=bar_width,
            bottom=bottom,
            color=colour,
            edgecolor="white",
            linewidth=0.4,
            label=label,
        )

        bottom += frequency

    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    axis.set_title(
        "Amarillo Wind Rose, 2020–2025\n"
        "Wind speed corrected from 100 m to 95 m hub height",
        pad=24,
    )
    axis.set_ylabel("Frequency (%)")
    axis.legend(
        title="Hub-height wind speed",
        loc="upper left",
        bbox_to_anchor=(1.05, 1.05),
    )

    figure.tight_layout()
    figure.savefig(
        os.path.join(SCRIPT_DIR, "wind_rose_2020_2025.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def make_layout_comparison_graph(layouts):
    rotor_diameter = vestas.rotor_diameter
    domain_size_d = COMPARISON_DOMAIN_D

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 6),
        sharex=True,
        sharey=True,
    )

    for axis, name, colour in zip(
        axes,
        ("grid", "staggered"),
        ("steelblue", "darkorange"),
    ):
        layout_d = layouts[name] / rotor_diameter

        axis.scatter(
            layout_d[:, 0],
            layout_d[:, 1],
            s=75,
            color=colour,
            edgecolor="black",
            linewidth=0.5,
        )

        for turbine_number, (x_position, y_position) in enumerate(
            layout_d
        ):
            axis.annotate(
                str(turbine_number),
                (x_position, y_position),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )

        axis.set_xlim(0.0, domain_size_d)
        axis.set_ylim(0.0, domain_size_d)
        axis.set_aspect("equal")
        axis.set_title(f"{name.title()} layout — 5D spacing")
        axis.set_xlabel("X position (rotor diameters)")
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Y position (rotor diameters)")

    figure.suptitle(
        "Baseline Layouts Inside the 25D × 25D Search Domain"
    )
    figure.tight_layout()
    figure.savefig(
        os.path.join(
            SCRIPT_DIR,
            "grid_vs_staggered_layouts.png",
        ),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def print_layout_results(name, metrics):
    print(f"\n===== {name.upper()} LAYOUT =====")
    print(f"AEP: {metrics['aep']:,.1f} MWh/year")
    print(f"LCOE: ${metrics['lcoe']:,.2f}/MWh")
    print(f"Wake loss: {metrics['wake_loss']:.2f}%")
    print(
        f"Maximum fatigue proxy: "
        f"{metrics['maximum_fatigue']:.3e}"
    )
    print(
        f"Mean fatigue proxy: "
        f"{metrics['mean_fatigue']:.3e}"
    )
    print(f"Worst turbine: {metrics['worst_turbine']}")
    print(
        f"Cable length: "
        f"{metrics['cable_length_m']:,.1f} m"
    )
    print(
        f"Road length: "
        f"{metrics['road_length_m']:,.1f} m"
    )


def run_experiment():
    print(
        "Loading synchronized 2020–2025 wind, "
        "temperature and pressure data..."
    )

    weather = load_real_weather_data()

    speeds = weather["wind_speed_m_s"].to_numpy()
    directions = weather["wind_direction_deg"].to_numpy()
    densities = weather["air_density_kg_m3"].to_numpy()

    layouts = {
        "grid": centre_layout_in_domain(grid_layout()),
        "staggered": centre_layout_in_domain(staggered_layout()),
    }

    results = {}

    for name, layout in layouts.items():
        performance = calculate_layout_performance(
            layout,
            speeds,
            directions,
            air_densities=densities,
        )

        lcoe = calculate_lcoe_details(
            layout,
            performance["aep"],
        )

        results[name] = {
            **performance,
            **lcoe,
        }

        print_layout_results(name, results[name])

    make_metrics_comparison_graph(results)
    make_wake_loss_graph(results)
    make_wind_rose_graph(speeds, directions)
    make_layout_comparison_graph(layouts)

    print(
        "\nGraphs saved:\n"
        "1. grid_vs_staggered_metrics.png\n"
        "2. grid_vs_staggered_wake_loss.png\n"
        "3. wind_rose_2020_2025.png\n"
        "4. grid_vs_staggered_layouts.png"
    )

    return results

if __name__ == "__main__":
    run_experiment()
