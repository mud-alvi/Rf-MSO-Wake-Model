
import matplotlib.pyplot as plt
import numpy as np
import math

import lcoe_model as lcoe
from main import calculate_layout_performance, load_real_weather_data
from turbine import vestas


# ===== EDITABLE SETTINGS =====
TURBINE_COUNT = 60
SPACING_D = 2.7
TURBINES_PER_COLUMN = 5
MINIMUM_DOMAIN_D = 25.0
# =============================


def make_staggered_layout(turbine_count):
    if turbine_count < 1:
        raise ValueError("Turbine count must be at least 1.")

    columns = min(TURBINES_PER_COLUMN, turbine_count)
    rows = math.ceil(turbine_count / columns)

    domain_d = MINIMUM_DOMAIN_D
    edge_margin_d = 0.5  # Keeps the complete rotor inside the site
    usable_d = domain_d - 2.0 * edge_margin_d

    x_spacing_d = usable_d / (rows - 1) if rows > 1 else 0.0

    # Odd rows are shifted by half of the Y spacing.
    y_denominator = columns - 0.5 if rows > 1 else columns - 1
    y_spacing_d = usable_d / y_denominator if y_denominator > 0 else 0.0

    layout_d = []

    for index in range(turbine_count):
        row = index // columns
        column = index % columns

        x = edge_margin_d + row * x_spacing_d
        y = edge_margin_d + column * y_spacing_d

        if row % 2:
            y += 0.5 * y_spacing_d

        layout_d.append((x, y))

    layout_d = np.asarray(layout_d, dtype=float)

    # Confirm the 4D minimum centre-to-centre spacing.
    for i in range(len(layout_d)):
        for j in range(i + 1, len(layout_d)):
            separation_d = np.linalg.norm(layout_d[i] - layout_d[j])

            if separation_d < SPACING_D - 1e-9:
                raise ValueError(
                    f"{turbine_count} turbines cannot fit inside a "
                    f"{domain_d}D × {domain_d}D site with minimum "
                    f"{SPACING_D}D spacing. Closest spacing: "
                    f"{separation_d:.2f}D."
                )

    layout = layout_d * vestas.rotor_diameter
    domain = np.array([domain_d, domain_d]) * vestas.rotor_diameter

    return layout, domain


def save_layout_graph(layout, domain, fatigue, results):
    diameter = vestas.rotor_diameter

    figure, axis = plt.subplots(figsize=(9, 7))
    points = axis.scatter(
        layout[:, 0] / diameter,
        layout[:, 1] / diameter,
        c=fatigue,
        s=90,
        cmap="viridis",
    )

    axis.set_xlim(0, domain[0] / diameter)
    axis.set_ylim(0, domain[1] / diameter)
    axis.set_aspect("equal")
    axis.set_xlabel("X position (D)")
    axis.set_ylabel("Y position (D)")
    axis.grid(alpha=0.25)
    axis.set_title(
        "Staggered layout 60 turbines\n"
        f"AEP {results['aep']:,.0f} MWh | "
        f"LCOE ${results['lcoe']:.2f}/MWh | "
        f"Max fatigue {results['maximum_fatigue']:.2e}"
    )

    colorbar = figure.colorbar(points, ax=axis)
    colorbar.set_label("Relative fatigue index")

    figure.tight_layout()
    figure.savefig("staggered_baseline.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def run(turbine_count):
    print("Loading synchronized wind, temperature and pressure data...")
    weather = load_real_weather_data()

    layout, domain = make_staggered_layout(turbine_count)

    performance = calculate_layout_performance(
        layout,
        weather["wind_speed_m_s"].to_numpy(),
        weather["wind_direction_deg"].to_numpy(),
        weather["air_density_kg_m3"].to_numpy(),
    )

    # Scale the existing LCOE model to the requested farm capacity.
    plant_capacity_kw = turbine_count * vestas.rated_power_kw
    lcoe.BASE_CAPEX_USD = lcoe.CAPEX_TOTAL_PER_KW * plant_capacity_kw
    lcoe.ANNUAL_OPEX_USD = lcoe.OPEX_PER_KW_YEAR * plant_capacity_kw

    costs = lcoe.calculate_lcoe_details(layout, performance["aep"])
    results = {**performance, **costs}

    print("\n===== STAGGERED BASELINE =====")
    print(f"Turbines: {turbine_count}")
    print(f"Plant capacity: {plant_capacity_kw / 1000:,.1f} MW")
    print(f"AEP: {results['aep']:,.1f} MWh/year")
    print(f"LCOE: ${results['lcoe']:.2f}/MWh")
    print(f"Wake loss: {results['wake_loss']:.2f}%")
    print(f"Maximum fatigue: {results['maximum_fatigue']:.3e}")
    print(f"Mean fatigue: {results['mean_fatigue']:.3e}")
    print(f"Worst turbine: {results['worst_turbine']}")
    print(f"Cable length: {results['cable_length_m']:,.1f} m")
    print(f"Cable cost: ${results['cable_cost_usd']:,.2f}")
    print(f"Road length: {results['road_length_m']:,.1f} m")
    print(f"Road cost: ${results['road_cost_usd']:,.2f}")
    print(f"Base CAPEX: ${lcoe.BASE_CAPEX_USD:,.2f}")
    print(f"Annual OPEX: ${lcoe.ANNUAL_OPEX_USD:,.2f}")
    print(f"Total CAPEX: ${results['total_capex']:,.2f}")

    save_layout_graph(
        layout,
        domain,
        results["per_turbine_fatigue"],
        results,
    )
    print("\nGraph saved: staggered_baseline.png")


if __name__ == "__main__":
    run(TURBINE_COUNT)