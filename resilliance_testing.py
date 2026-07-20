import os
import pandas as pd
import matplotlib.pyplot as plt
import main
 
NOAA_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "4356691.csv")
 
# Thresholds (editable)
HEAT_F, COLD_F, GUST_MPH = 104, -4, 55
 
# Scenario penalties: fraction of that day's production lost
SCENARIOS = {
    "Mild weather":      {"heat": 0.00, "cold": 0.00, "gust": 0.10},
    "Moderate weather": {"heat": 0.05, "cold": 0.10, "gust": 0.50},
    "Extreme weather":     {"heat": 0.10, "cold": 0.25, "gust": 1.00},
}
 
 
def load_noaa(start_year=main.START_YEAR, end_year=main.END_YEAR):
    df = pd.read_csv(NOAA_CSV)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df[df["DATE"].dt.year.between(start_year, end_year)].copy()
    df["heat"] = df["TMAX"] >= HEAT_F
    df["cold"] = df["TMIN"] <= COLD_F
    df["gust"] = df["WSF5"] >= GUST_MPH
    return df
 
 
def weather_loss_fraction(df, penalties):
    # Worst single penalty per day (avoids stacking heat+cold+gust on one day)
    daily = pd.concat(
        [df["heat"] * penalties["heat"],
         df["cold"] * penalties["cold"],
         df["gust"] * penalties["gust"]],
        axis=1,
    ).max(axis=1)
    return daily.mean()  # equal weight per day -- coarse, not energy-weighted
 
 
def run():
    wake = main.run_experiment()
    grid_aep, stag_aep = wake["grid_aep"], wake["stag_aep"]
 
    noaa = load_noaa()
    print(f"NOAA days: {len(noaa)} | heat={noaa['heat'].sum()} "
          f"cold={noaa['cold'].sum()} gust={noaa['gust'].sum()}")
    names, grid, stag = ["No weather loss"], [grid_aep], [stag_aep]
 
    for name, penalties in SCENARIOS.items():
        loss = weather_loss_fraction(noaa, penalties)
        g_adj, s_adj = grid_aep * (1 - loss), stag_aep * (1 - loss)
        improvement = (s_adj - g_adj) / g_adj * 100
 
        print(f"\n{name} scenario ({loss * 100:.1f}% weather loss)")
        print(f"  Grid:      {grid_aep:,.0f} -> {g_adj:,.0f} MWh")
        print(f"  Staggered: {stag_aep:,.0f} -> {s_adj:,.0f} MWh")
        print(f"  Staggered-vs-grid improvement: {improvement:+.2f}%")
        names.append(name)
        grid.append(g_adj)
        stag.append(s_adj)
    
    x = range(len(names))

    grid_bars = plt.bar(
        [i - 0.2 for i in x], grid, 0.4, label="Grid"
    )
    stag_bars = plt.bar(
        [i + 0.2 for i in x], stag, 0.4, label="Staggered"
    )

    plt.xticks(list(x), names)
    plt.ylabel("AEP (MWh) — truncated axis")

    plt.title(
        f"Average Annual Wind-Farm Energy Output ({main.START_YEAR}-{main.END_YEAR})\n"
        "Extreme heat, cold, and high-wind losses"
    )

    lowest = min(grid + stag)
    highest = max(grid + stag)
    margin = (highest - lowest) * 0.2

    plt.ylim(lowest - margin, highest + margin)

    plt.bar_label(grid_bars, fmt="%.0f", padding=3, fontsize=8)
    plt.bar_label(stag_bars, fmt="%.0f", padding=3, fontsize=8)

    plt.legend()
    plt.grid(axis="y", alpha=0.25)

    plt.figtext(
        0.5,
        0.01,
        "Y-axis does not start at zero.(for emphasis on differences between scenarios)",
        ha="center",
        fontsize=8,
        style="italic",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    plt.savefig(
        os.path.join(
            os.path.dirname(__file__),
            "graph5_noaa_aep.png"
        ),
        dpi=150,
    )

    plt.close()


 
 
if __name__ == "__main__":
    run()