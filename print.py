import matplotlib.pyplot as plt

layouts = ["Grid", "Staggered", "Seed 1 Winner"]
colors = ["steelblue", "darkorange", "seagreen"]

# Performance results
aep = [
    154353.0,  # Grid
    154447.3,  # Staggered
    156231.2   # Seed 1 winner
]

lcoe = [
    57.70,  # Grid
    57.72,  # Staggered
    56.94   # Seed 1 winner
]

# Actual maximum fatigue values
max_fatigue_actual = [
    6.453e29,  # Grid, estimated from staggered relative index
    6.440e29,  # Staggered baseline
    5.035e29   # Seed 1 winner
]

# Normalize fatigue relative to grid
max_fatigue = [
    value / max_fatigue_actual[0]
    for value in max_fatigue_actual
]

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

fig.suptitle(
    "Grid vs Staggered vs GA Winner — Amarillo 2020–2025",
    fontsize=17,
    fontweight="bold"
)

# AEP graph
bars = axes[0].bar(layouts, aep, color=colors)

axes[0].set_title("Annual Energy Production")
axes[0].set_ylabel("AEP (MWh/year)")
axes[0].grid(axis="y", alpha=0.3)

for bar, value in zip(bars, aep):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:,.0f}",
        ha="center",
        va="bottom"
    )

# LCOE graph
bars = axes[1].bar(layouts, lcoe, color=colors)

axes[1].set_title("Levelized Cost of Energy")
axes[1].set_ylabel("LCOE ($/MWh)")
axes[1].grid(axis="y", alpha=0.3)

for bar, value in zip(bars, lcoe):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"${value:.2f}",
        ha="center",
        va="bottom"
    )

# Maximum fatigue graph
bars = axes[2].bar(layouts, max_fatigue, color=colors)

axes[2].set_title("Maximum Fatigue Proxy")
axes[2].set_ylabel("Relative Fatigue Index (Grid = 1.00)")
axes[2].axhline(
    1.0,
    color="dimgray",
    linestyle="--",
    linewidth=1.5
)
axes[2].set_ylim(0, 1.1)
axes[2].grid(axis="y", alpha=0.3)

for bar, value in zip(bars, max_fatigue):
    axes[2].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

# Improve x-axis labels
for ax in axes:
    ax.tick_params(axis="x", rotation=10)

plt.tight_layout()

# Save the graph as a high-resolution image
plt.savefig(
    "grid_vs_staggered_vs_seed1_winner.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()