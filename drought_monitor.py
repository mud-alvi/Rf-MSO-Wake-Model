import os

import pandas as pd


# Drought CSV stored beside this Python file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DROUGHT_CSV = os.path.join(SCRIPT_DIR, "Drought_AM_TX_2020-2025.csv")

# U.S. Drought Monitor categories, from no drought to exceptional drought
DROUGHT_CATEGORIES = ["None", "D0", "D1", "D2", "D3", "D4"]
DROUGHT_WEIGHTS = {"None": 0, "D0": 1, "D1": 2, "D2": 3, "D3": 4, "D4": 5}


def load_drought_data(
    csv_path=DROUGHT_CSV,
    start_year=2020,
    end_year=2025,
    county_fips="48375", # Potter County, TX
):
    """Load Potter County drought data and expand each week into daily rows."""
    df = pd.read_csv(csv_path)

    # Confirm that the downloaded file has the required columns
    required = {
        "FIPS", "County", "State", "ValidStart", "ValidEnd",
        *DROUGHT_CATEGORIES,
    }
    # to confirm nothing is missing, checking the difference
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Drought CSV is missing columns: {sorted(missing)}")

    # Keep only the selected county
    df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)
    df = df[df["FIPS"] == str(county_fips)].copy()
    if df.empty:
        raise ValueError(f"No drought rows found for county FIPS {county_fips}.")

    # Dates in this CSV use day/month/year format
    df["ValidStart"] = pd.to_datetime(
        df["ValidStart"], format="%d/%m/%Y", errors="raise"
    )
    df["ValidEnd"] = pd.to_datetime(
        df["ValidEnd"], format="%d/%m/%Y", errors="raise"
    )

    start_date = pd.Timestamp(start_year, 1, 1)
    end_date = pd.Timestamp(end_year, 12, 31)

    # Keep weeks that overlap the requested date range
    df = df[
        (df["ValidEnd"] >= start_date) & (df["ValidStart"] <= end_date)
    ].copy()

    # Validate the county-area percentages
    area_total = df[DROUGHT_CATEGORIES].sum(axis=1)
    if not area_total.between(99.9, 100.1).all():
        raise ValueError("Drought percentages do not add to approximately 100%.")

    # Area-weighted severity: 0 means no drought and 5 means all area is D4
    df["drought_severity"] = sum(
        df[category] * weight
        for category, weight in DROUGHT_WEIGHTS.items()
    ) / 100.0
    df["drought_area_pct"] = 100.0 - df["None"]
    df["dominant_category"] = df[DROUGHT_CATEGORIES].idxmax(axis=1)

    # Expand every weekly record across its seven valid days
    df["DATE"] = df.apply(
        lambda row: pd.date_range(row["ValidStart"], row["ValidEnd"], freq="D"),
        axis=1,
    )
    daily = df.explode("DATE", ignore_index=True)
    daily = daily[daily["DATE"].between(start_date, end_date)].copy()

    # One drought observation should exist for every calendar day
    daily = daily.sort_values("DATE").drop_duplicates("DATE", keep="last")
    expected_days = len(pd.date_range(start_date, end_date, freq="D"))
    if len(daily) != expected_days:
        raise ValueError(
            f"Drought data covers {len(daily)} days; expected {expected_days}."
        )

    keep = [
        "DATE", "FIPS", "County", "State", *DROUGHT_CATEGORIES,
        "drought_area_pct", "drought_severity", "dominant_category",
    ]
    return daily[keep].reset_index(drop=True)

# Return a summary of the average county-area percentage in each drought category by year
def yearly_drought_summary(daily):
    summary = daily.copy()
    summary["Year"] = summary["DATE"].dt.year
    return summary.groupby("Year")[DROUGHT_CATEGORIES].mean()

# Finally running the drought monitor code to print the summary and mean severity
if __name__ == "__main__":
    drought = load_drought_data()
    print(yearly_drought_summary(drought).round(2))
    print(f"\nMean drought severity: {drought['drought_severity'].mean():.2f}/5")