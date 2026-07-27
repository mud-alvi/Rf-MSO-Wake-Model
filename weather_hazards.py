import os

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HAZARD_CSV = os.path.join(
    SCRIPT_DIR,
    "humidity_precip_dust_lightning_2020_2025_full24h_NOAA_KAMA.csv",
)

REQUIRED_COLUMNS = [
    "time",
    "relative_humidity_pct",
    "relative_humidity_pct_filled",
    "precip_in",
    "precip_trace_flag",
    "precip_multihour_accum_flag",
    "dust_flag",
    "thunderstorm_flag",
]

FLAG_COLUMNS = [
    "precip_trace_flag",
    "precip_multihour_accum_flag",
    "dust_flag",
    "thunderstorm_flag",
]


def _to_nullable_boolean(series, column):
    # Preserve missing flags instead of treating them as False
    values = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    text = series.astype("string").str.strip().str.lower()
    invalid = text.notna() & ~text.isin(values)

    if invalid.any():
        examples = sorted(text[invalid].unique().tolist())
        raise ValueError(f"Invalid Boolean values in {column}: {examples}")

    return text.map(values).astype("boolean")


def load_hourly_hazard_data(
    csv_path=HAZARD_CSV,
    start_year=2020,
    end_year=2025,
):
    # Load and validate the synchronized NOAA KAMA hourly observations
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Hourly hazard CSV not found: {csv_path}\n"
            "Place the CSV in the same folder as weather_hazards.py."
        )

    data = pd.read_csv(csv_path)
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            "Hourly hazard CSV is missing columns: "
            + ", ".join(missing_columns)
        )

    data = data[REQUIRED_COLUMNS].copy()
    data["time_utc"] = pd.to_datetime(
        data.pop("time"),
        errors="coerce",
        utc=True,
    )
    if data["time_utc"].isna().any():
        bad_rows = data.index[data["time_utc"].isna()].tolist()[:10]
        raise ValueError(f"Invalid timestamps at CSV rows: {bad_rows}")

    if data["time_utc"].duplicated().any():
        duplicates = (
            data.loc[data["time_utc"].duplicated(), "time_utc"]
            .astype(str)
            .tolist()[:10]
        )
        raise ValueError(f"Duplicate hourly timestamps found: {duplicates}")

    data = data.sort_values("time_utc").reset_index(drop=True)
    expected = pd.date_range(
        data["time_utc"].iloc[0],
        data["time_utc"].iloc[-1],
        freq="h",
    )
    missing_hours = expected.difference(data["time_utc"])
    if len(missing_hours):
        examples = missing_hours.astype(str).tolist()[:10]
        raise ValueError(
            f"{len(missing_hours)} hourly timestamps are missing: {examples}"
        )

    numeric_columns = [
        "relative_humidity_pct",
        "relative_humidity_pct_filled",
        "precip_in",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    for column in FLAG_COLUMNS:
        data[column] = _to_nullable_boolean(data[column], column)

    data["time_local"] = data["time_utc"].dt.tz_convert("America/Chicago")
    local_year = data["time_local"].dt.year
    data = data[local_year.between(start_year, end_year)].copy()
    data["year"] = data["time_local"].dt.year

    if data.empty:
        raise ValueError(
            f"No Amarillo local-time rows found for {start_year}-{end_year}."
        )

    return data


def yearly_hazard_summary(data):
    # Summarize hazard exposure without assigning power-loss penalties
    rows = []

    for year, group in data.groupby("year", sort=True):
        humidity = group["relative_humidity_pct_filled"]
        precipitation = group["precip_in"]

        rows.append(
            {
                "year": int(year),
                "observation_hours": len(group),
                "avg_relative_humidity_pct": humidity.mean(),
                "max_relative_humidity_pct": humidity.max(),
                "recorded_precip_in": precipitation.sum(min_count=1),
                "measurable_precip_hours": (precipitation > 0).sum(),
                "trace_precip_hours": group[
                    "precip_trace_flag"
                ].sum(min_count=1),
                "multihour_accum_flags": group[
                    "precip_multihour_accum_flag"
                ].sum(min_count=1),
                "dust_event_hours": group["dust_flag"].sum(min_count=1),
                "thunderstorm_hours": group[
                    "thunderstorm_flag"
                ].sum(min_count=1),
                "humidity_valid_hours": humidity.notna().sum(),
                "humidity_missing_hours": humidity.isna().sum(),
                "precip_valid_hours": precipitation.notna().sum(),
                "precip_missing_hours": precipitation.isna().sum(),
            }
        )

    return pd.DataFrame(rows).set_index("year")


def make_hourly_hazard_graph(
    summary,
    output_path=None,
):
    # Plot annual exposure indicators; these are not modeled AEP losses
    if output_path is None:
        output_path = os.path.join(
            SCRIPT_DIR,
            "graph10_hourly_hazard_summary.png",
        )

    figure, graphs = plt.subplots(2, 2, figsize=(11, 8))
    years = summary.index.astype(str)

    panels = [
        (
            "avg_relative_humidity_pct",
            "Average relative humidity",
            "Relative humidity (%)",
            "steelblue",
        ),
        (
            "recorded_precip_in",
            "Recorded precipitation",
            "Precipitation (in)",
            "royalblue",
        ),
        (
            "dust_event_hours",
            "Dust-event hours",
            "Flagged hours",
            "darkorange",
        ),
        (
            "thunderstorm_hours",
            "Thunderstorm hours",
            "Flagged hours",
            "purple",
        ),
    ]

    for graph, (column, title, ylabel, color) in zip(
        graphs.flat,
        panels,
    ):
        values = summary[column].astype(float)
        bars = graph.bar(years, values, color=color)
        graph.set_title(title)
        graph.set_xlabel("Year")
        graph.set_ylabel(ylabel)
        graph.grid(axis="y", alpha=0.25)
        graph.margins(y=0.15)
        graph.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)

    figure.suptitle(
        "NOAA KAMA Hourly Hazard Exposure (2020-2025)\n"
        "Exposure indicators only; no AEP penalty is assigned",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _print_summary(data, summary):
    print("\nNOAA KAMA hourly hazard summary")
    print(summary.round(2).to_string())
    print("\nSix-year totals")
    print(f"  Observation hours: {len(data)}")
    print(
        "  Average filled relative humidity: "
        f"{data['relative_humidity_pct_filled'].mean():.2f}%"
    )
    print(
        "  Recorded precipitation: "
        f"{data['precip_in'].sum(min_count=1):.2f} in"
    )
    print(
        "  Measurable-precipitation hours: "
        f"{int((data['precip_in'] > 0).sum())}"
    )
    print(
        "  Trace-precipitation hours: "
        f"{int(data['precip_trace_flag'].sum(min_count=1))}"
    )
    print(
        "  Dust-event hours: "
        f"{int(data['dust_flag'].sum(min_count=1))}"
    )
    print(
        "  Thunderstorm hours: "
        f"{int(data['thunderstorm_flag'].sum(min_count=1))}"
    )


if __name__ == "__main__":
    hourly = load_hourly_hazard_data()
    annual = yearly_hazard_summary(hourly)
    _print_summary(hourly, annual)
    saved_graph = make_hourly_hazard_graph(annual)
    print(f"\nSaved {os.path.basename(saved_graph)}")
