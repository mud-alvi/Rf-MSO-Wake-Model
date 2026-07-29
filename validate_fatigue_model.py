import numpy as np

from layouts import grid_layout, staggered_layout
from main import (
    AMBIENT_TI,
    calculate_farm_fatigue,
)
from turbine import vestas
from wake_model import (
    effective_turbulence_intensity,
    tower_base_load_proxy_at_point,
)

D = vestas.rotor_diameter


def case_header(text):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def main():
    wind_speeds = np.full(4, 8.0)
    wind_directions = np.zeros(4)
    air_densities = np.full(4, 1.225)

    case_header("1) Single isolated turbine should have ambient turbulence only")
    isolated_positions = [(0.0, 0.0)]
    ti = effective_turbulence_intensity(isolated_positions, 0.0, 0.0, 8.0, ambient_ti=AMBIENT_TI)
    print(f"Effective TI for isolated turbine = {ti:.5f}")
    assert np.isclose(ti, AMBIENT_TI, atol=1e-8), "Isolated turbine TI must equal ambient TI"

    case_header("2) Identical isolated turbines should produce identical fatigue")
    far_apart_positions = [(0.0, 0.0), (0.0, 1000.0 * D)]
    fatigue_result = calculate_farm_fatigue(
        far_apart_positions,
        wind_speeds,
        wind_directions,
        air_densities=air_densities,
    )
    print(f"Per-turbine fatigue = {fatigue_result['per_turbine_fatigue']}")
    assert np.allclose(
        fatigue_result["per_turbine_fatigue"],
        fatigue_result["per_turbine_fatigue"][0],
        rtol=1e-8,
        atol=1e-12,
    ), "Identical isolated turbines must have identical fatigue"

    case_header("3) Downstream turbine should have higher fatigue than unwaked turbine")
    wake_positions = [(0.0, 0.0), (5.0 * D, 0.0), (0.0, 1000.0 * D)]
    wake_fatigue = calculate_farm_fatigue(
        wake_positions,
        wind_speeds,
        wind_directions,
        air_densities=air_densities,
    )
    print(f"Per-turbine fatigue = {wake_fatigue['per_turbine_fatigue']}")
    assert wake_fatigue["per_turbine_fatigue"][1] > wake_fatigue["per_turbine_fatigue"][2], (
        "Downstream turbine must have higher fatigue than a non-waked turbine"
    )

    case_header("4) Increasing spacing should reduce wake-added fatigue")
    close_positions = [(0.0, 0.0), (5.0 * D, 0.0)]
    far_positions = [(0.0, 0.0), (10.0 * D, 0.0)]
    close_fatigue = calculate_farm_fatigue(
        close_positions,
        wind_speeds,
        wind_directions,
        air_densities=air_densities,
    )
    far_fatigue = calculate_farm_fatigue(
        far_positions,
        wind_speeds,
        wind_directions,
        air_densities=air_densities,
    )
    print(f"Max fatigue at 5D = {close_fatigue['maximum_fatigue']:.6e}")
    print(f"Max fatigue at 10D = {far_fatigue['maximum_fatigue']:.6e}")
    assert far_fatigue["maximum_fatigue"] < close_fatigue["maximum_fatigue"], (
        "Increasing spacing should reduce maximum fatigue"
    )

    case_header("5) Rotating wind direction should change the worst turbine")
    rotated_positions = [(0.0, 0.0), (5.0 * D, 0.0), (0.0, 5.0 * D)]
    fatigue_0 = calculate_farm_fatigue(
        rotated_positions,
        wind_speeds,
        np.zeros(4),
        air_densities=air_densities,
    )
    fatigue_90 = calculate_farm_fatigue(
        rotated_positions,
        wind_speeds,
        np.full(4, 90.0),
        air_densities=air_densities,
    )
    print(f"Worst turbine at 0° = {fatigue_0['worst_turbine']}")
    print(f"Worst turbine at 90° = {fatigue_90['worst_turbine']}")
    assert fatigue_0["worst_turbine"] != fatigue_90["worst_turbine"], (
        "Rotating wind direction must change which turbine is worst"
    )

    case_header("6) Constant temperature and pressure should reproduce constant density")
    default_fatigue = calculate_farm_fatigue(
        close_positions,
        wind_speeds,
        wind_directions,
    )
    constant_density_fatigue = calculate_farm_fatigue(
        close_positions,
        wind_speeds,
        wind_directions,
        air_densities=air_densities,
    )
    assert np.allclose(
        default_fatigue["per_turbine_fatigue"],
        constant_density_fatigue["per_turbine_fatigue"],
        rtol=1e-8,
        atol=1e-12,
    ), "Constant temperature and pressure must reproduce constant density behaviour"

    case_header("7) Compare grid versus staggered layouts")
    grid_fatigue = calculate_farm_fatigue(
        grid_layout(),
        wind_speeds,
        wind_directions,
        air_densities=air_densities,
    )
    staggered_fatigue = calculate_farm_fatigue(
        staggered_layout(),
        wind_speeds,
        wind_directions,
        air_densities=air_densities,
    )
    print(
        f"Grid max fatigue = {grid_fatigue['maximum_fatigue']:.6e}, "
        f"Staggered max fatigue = {staggered_fatigue['maximum_fatigue']:.6e}"
    )

    case_header("8) TI sensitivity sweep")
    for ti in [0.06, 0.077, 0.10, 0.14]:
        ti_fatigue = calculate_farm_fatigue(
            close_positions,
            wind_speeds,
            wind_directions,
            air_densities=air_densities,
            ambient_ti=ti,
        )
        print(
            f"TI={ti:.3f}: max={ti_fatigue['maximum_fatigue']:.6e}, "
            f"mean={ti_fatigue['mean_fatigue']:.6e}"
        )

    case_header("9) Exponent sensitivity sweep")
    for exponent in [3.0, 4.0, 5.0]:
        exp_fatigue = calculate_farm_fatigue(
            close_positions,
            wind_speeds,
            wind_directions,
            air_densities=air_densities,
            exponent=exponent,
        )
        print(
            f"m={exponent:.0f}: max={exp_fatigue['maximum_fatigue']:.6e}, "
            f"mean={exp_fatigue['mean_fatigue']:.6e}"
        )

    print("\nAll validation checks passed.")


if __name__ == "__main__":
    main()
