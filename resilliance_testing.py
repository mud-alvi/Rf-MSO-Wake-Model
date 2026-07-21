import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import main
from drought_monitor import load_drought_data


# NOAA file stored in the same folder as this code
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NOAA_CSV = os.path.join(SCRIPT_DIR, "4356691.csv")

# Extreme-weather thresholds
HEAT_F = 104
COLD_F = -4
GUST_MPH = 55

# Assumed production loss when an extreme-weather event occurs
LOSS_LEVELS = {
    "Low": {"heat": 0.00, "cold": 0.00, "gust": 0.10},
    "Moderate": {"heat": 0.05, "cold": 0.10, "gust": 0.50},
    "High": {"heat": 0.10, "cold": 0.25, "gust": 1.00},
}


def load_noaa(start_year=main.START_YEAR, end_year=main.END_YEAR):
    # Load NOAA data and mark extreme-weather days
    df = pd.read_csv(NOAA_CSV)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df[df["DATE"].dt.year.between(start_year, end_year)].copy()

    # True means that the weather threshold was reached on that day
    df["heat"] = df["TMAX"] >= HEAT_F
    df["cold"] = df["TMIN"] <= COLD_F
    df["gust"] = df["WSF5"] >= GUST_MPH
    return df


def weather_loss_fraction(df, penalties):
    # Calculate the average assumed production loss across all days
    daily = pd.concat(
        [
            df["heat"] * penalties["heat"],
            df["cold"] * penalties["cold"],
            df["gust"] * penalties["gust"],
        ],
        axis=1,
    ).max(axis=1)

    # The worst single loss is used if hazards overlap on the same day
    return daily.mean()


def penalties_for_graph(hazard, level):
    # Keep only the selected hazard, or keep all for total weather
    penalties = LOSS_LEVELS[level]

    if hazard == "total":
        return penalties

    return {
        "heat": penalties["heat"] if hazard == "heat" else 0.0,
        "cold": penalties["cold"] if hazard == "cold" else 0.0,
        "gust": penalties["gust"] if hazard == "gust" else 0.0,
    }


def make_graph(df, grid_aep, stag_aep, hazard, title, filename):
    # Create one graph containing baseline, low, moderate, and high cases
    names = ["No weather loss", "Low", "Moderate", "High"]
    grid = [grid_aep]
    staggered = [stag_aep]

    # Calculate the three loss levels for this graph
    for level in ["Low", "Moderate", "High"]:
        penalties = penalties_for_graph(hazard, level)
        loss = weather_loss_fraction(df, penalties)

        grid.append(grid_aep * (1 - loss))
        staggered.append(stag_aep * (1 - loss))

        print(f"  {level}: {loss * 100:.2f}% average AEP loss")

    x = range(len(names))
    figure, graph = plt.subplots(figsize=(9, 6))

    grid_bars = graph.bar(
        [i - 0.2 for i in x], grid, 0.4, label="Grid"
    )
    stag_bars = graph.bar(
        [i + 0.2 for i in x], staggered, 0.4, label="Staggered"
    )

    graph.set_xticks(list(x))
    graph.set_xticklabels(names)
    graph.set_ylabel("Average annual AEP (MWh) - truncated axis")
    graph.set_title(
        f"{title} ({main.START_YEAR}-{main.END_YEAR})\n"
        "Low, moderate, and high are assumed production-loss levels"
    )

    # Shorten the y-axis so the small scenario differences are visible
    lowest = min(grid + staggered)
    highest = max(grid + staggered)
    margin = max((highest - lowest) * 0.20, highest * 0.005)
    graph.set_ylim(max(0, lowest - margin), highest + margin)

    graph.bar_label(grid_bars, fmt="%.0f", padding=3, fontsize=8)
    graph.bar_label(stag_bars, fmt="%.0f", padding=3, fontsize=8)
    graph.legend()
    graph.grid(axis="y", alpha=0.25)

    figure.text(
        0.5,
        0.01,
        "Y-axis does not start at zero. Values are scenario assumptions, "
        "not measured turbine damage.",
        ha="center",
        fontsize=8,
        style="italic",
    )

    figure.tight_layout(rect=[0, 0.04, 1, 1])
    figure.savefig(os.path.join(SCRIPT_DIR, filename), dpi=150)
    plt.close(figure)


def make_drought_resilience_graph(noaa):
    # Load drought data for the same years as the NOAA data
    drought = load_drought_data(
        start_year=main.START_YEAR,
        end_year=main.END_YEAR,
    )

    # Match each weather day with its drought severity
    combined = pd.merge(
        noaa,
        drought[["DATE", "drought_severity"]],
        on="DATE",
        how="inner",
    )

    # Use the existing high-loss assumptions for the resilience comparison
    high = LOSS_LEVELS["High"]
    combined["weather_loss"] = pd.concat(
        [
            combined["heat"] * high["heat"],
            combined["cold"] * high["cold"],
            combined["gust"] * high["gust"],
        ],
        axis=1,
    ).max(axis=1)

    # Resilience is the percentage of modeled production that remains
    combined["weather_resilience"] = 100 * (1 - combined["weather_loss"])
    combined["month"] = combined["DATE"].dt.to_period("M")

    # Average the daily values for each month
    monthly = combined.groupby("month").agg(
        drought_severity=("drought_severity", "mean"),
        weather_resilience=("weather_resilience", "mean"),
    )

    correlation = monthly["drought_severity"].corr(
        monthly["weather_resilience"]
    )

    figure, graph = plt.subplots(figsize=(9, 6))
    graph.scatter(
        monthly["drought_severity"],
        monthly["weather_resilience"],
        color="purple",
        alpha=0.75,
    )

    # Add a simple trend line to make the relationship easier to see
    line = np.polyfit(
        monthly["drought_severity"],
        monthly["weather_resilience"],
        1,
    )
    x_line = np.linspace(
        monthly["drought_severity"].min(),
        monthly["drought_severity"].max(),
        100,
    )
    graph.plot(x_line, line[0] * x_line + line[1], color="black")

    graph.set_xlabel("Monthly drought severity (0 = none, 5 = exceptional)")
    graph.set_ylabel("Modeled weather resilience (%)")
    graph.set_title(
        f"Drought and Weather Resilience ({main.START_YEAR}-{main.END_YEAR})\n"
        f"Monthly comparison, correlation = {correlation:.2f}"
    )
    graph.grid(alpha=0.25)

    figure.text(
        0.5,
        0.01,
        "Higher resilience means more modeled energy production remains. "
        "Drought is compared only; it is not an extra AEP penalty.",
        ha="center",
        fontsize=8,
        style="italic",
    )

    figure.tight_layout(rect=[0, 0.05, 1, 1])
    filename = "graph9_drought_resilience.png"
    figure.savefig(os.path.join(SCRIPT_DIR, filename), dpi=150)
    plt.close(figure)

    print(f"\nDrought-resilience correlation: {correlation:.2f}")
    print(
        "  This graph checks whether worse drought months also had less "
        "modeled weather production remaining."
    )
    print("  This shows an association, not proof that drought causes AEP loss.")
    print(f"  Saved {filename}")


def run():
    # Run the wake model once and create the resilience graphs
    wake = main.run_experiment()
    grid_aep = wake["grid_aep"]
    stag_aep = wake["stag_aep"]
    noaa = load_noaa()

    print(
        f"NOAA days: {len(noaa)} | heat={noaa['heat'].sum()} "
        f"cold={noaa['cold'].sum()} gust={noaa['gust'].sum()}"
    )

    # Explain resilience in simple words
    print(
        "\nWeather resilience means how much normal wind-farm energy "
        "production remains during extreme weather."
    )

    graphs = [
        ("heat", "Extreme-Heat Resilience", "graph5_heat_resilience.png"),
        ("cold", "Extreme-Cold Resilience", "graph6_cold_resilience.png"),
        ("gust", "High-Gust Resilience", "graph7_gust_resilience.png"),
        (
            "total",
            "Combined Weather Resilience",
            "graph8_total_weather_resilience.png",
        ),
    ]

    for hazard, title, filename in graphs:
        print(f"\n{title}")
        make_graph(
            noaa,
            grid_aep,
            stag_aep,
            hazard,
            title,
            filename,
        )
        print(f"  Saved {filename}")

    # Create the separate drought-resilience relationship graph
    make_drought_resilience_graph(noaa)


if __name__ == "__main__":
    run()