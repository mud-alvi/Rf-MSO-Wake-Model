"""Editable multi-start GA for AEP, LCOE, wake steering and fatigue."""

import os

import matplotlib.pyplot as plt
import numpy as np

from layouts import layout_families, staggered_layout
from lcoe_model import calculate_lcoe_details
from main import (
    build_wind_rose_cases,
    calculate_layout_performance,
    load_real_weather_data,
    _active_yaw_angles,
    rotate_layout_to_wind_frame,
    wind_speed_at_hub,
)
from turbine import vestas

# ========================= EXPERIMENT SETTINGS =========================
# Peers can edit these values directly in their IDE for separate runs.
SEEDS = [1, 8, 42]
POPULATION_SIZE = 60
GENERATIONS = 20
TOURNAMENT_SIZE = 4
ELITE_COUNT = 3
FULL_CHECK_COUNT = 3

SEARCH_WIDTH_D = 25
SEARCH_HEIGHT_D = 25
MIN_SPACING_D = 4
DIRECTION_BINS = 16
SPEED_BINS = 8

INITIAL_MUTATION_RATE = 0.30
FINAL_MUTATION_RATE = 0.10
INITIAL_MUTATION_DISTANCE_D = 0.90
FINAL_MUTATION_DISTANCE_D = 0.35
STAGNATION_LIMIT = 4

YAW_MIN_DEG = -25.0
YAW_MAX_DEG = 25.0
WAKE_LOSS_PENALTY = 0.05
MEAN_FATIGUE_PENALTY = 0.05
SAVE_GRAPHS = True
# ======================================================================

D = vestas.rotor_diameter
TURBINES = 30
WIDTH = SEARCH_WIDTH_D * D
HEIGHT = SEARCH_HEIGHT_D * D
MIN_SPACING = MIN_SPACING_D * D
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def sort_layout(layout):
    layout = np.asarray(layout, dtype=float)
    return layout[np.lexsort((layout[:, 1], layout[:, 0]))]


def centre_in_domain(layout):
    layout = np.asarray(layout, dtype=float).copy()
    layout -= layout.min(axis=0)
    extent = layout.max(axis=0)
    layout += np.array([(WIDTH - extent[0]) / 2.0, (HEIGHT - extent[1]) / 2.0])
    return layout


def generate_layout(rng):
    layout = []
    for _ in range(100000):  #30 turbine change 2 
        candidate = rng.uniform([0.0, 0.0], [WIDTH, HEIGHT])
        if all(
            np.linalg.norm(candidate - existing) >= MIN_SPACING
            for existing in layout
        ):
            layout.append(candidate)
            if len(layout) == TURBINES:
                return sort_layout(layout)
    raise RuntimeError("Could not generate a valid layout in the search domain.")


def repair_layout(layout, rng):
    repaired = []
    for original in np.asarray(layout, dtype=float):
        candidate = np.clip(original, [0.0, 0.0], [WIDTH, HEIGHT])
        for attempt in range(10000):
            if all(
                np.linalg.norm(candidate - existing) >= MIN_SPACING
                for existing in repaired
            ):
                repaired.append(candidate)
                break
            if attempt < 100:
                candidate = np.clip(
                    original + rng.normal(0.0, 0.35 * D, 2),
                    [0.0, 0.0],
                    [WIDTH, HEIGHT],
                )
            else:
                candidate = rng.uniform([0.0, 0.0], [WIDTH, HEIGHT])
        else:
            return generate_layout(rng)
    return sort_layout(repaired)


def make_individual(layout, yaw=None):
    return {
        "layout": np.asarray(layout, dtype=float),
        "yaw": (
            np.zeros(DIRECTION_BINS)
            if yaw is None
            else np.asarray(yaw, dtype=float)
        ),
    }


def copy_individual(individual):
    return {
        "layout": individual["layout"].copy(),
        "yaw": individual["yaw"].copy(),
    }


def mutate_individual(individual, rng, rate, distance):
    child = copy_individual(individual)
    position_mask = rng.random(TURBINES) < rate
    child["layout"][position_mask] += rng.normal(
        0.0, distance, (position_mask.sum(), 2)
    )
    yaw_mask = rng.random(DIRECTION_BINS) < rate
    child["yaw"][yaw_mask] += rng.normal(0.0, 4.0, yaw_mask.sum())
    child["yaw"] = np.clip(child["yaw"], YAW_MIN_DEG, YAW_MAX_DEG)
    child["layout"] = repair_layout(child["layout"], rng)
    return child


def generate_population(size, rng):
    families = [
        make_individual(repair_layout(centre_in_domain(layout), rng))
        for layout in layout_families(rng)
    ]
    population = families[:size]
    while len(population) < size:
        parent = population[rng.integers(len(population))]
        population.append(
            mutate_individual(parent, rng, 0.35, 0.75 * D)
        )
    return population


def tournament_selection(population, scores, rng):
    competitors = rng.choice(
        len(population),
        size=min(TOURNAMENT_SIZE, len(population)),
        replace=False,
    )
    winner = max(competitors, key=lambda index: scores[index])
    return copy_individual(population[winner])


def crossover(parent_1, parent_2, rng):
    position_mask = rng.random(TURBINES) < 0.5
    yaw_mask = rng.random(DIRECTION_BINS) < 0.5
    child_layout = parent_2["layout"].copy()
    child_yaw = parent_2["yaw"].copy()
    child_layout[position_mask] = parent_1["layout"][position_mask]
    child_yaw[yaw_mask] = parent_1["yaw"][yaw_mask]
    return make_individual(child_layout, child_yaw)


def evaluate_individual(individual, speeds, directions, densities, weights):
    metrics = calculate_layout_performance(
        individual["layout"],
        speeds,
        directions,
        air_densities=densities,
        sample_weights=weights,
        yaw_schedule=individual["yaw"],
    )
    return {**metrics, **calculate_lcoe_details(individual["layout"], metrics["aep"])}


def is_feasible(metrics, baseline, tolerance=1e-9):
    return (
        metrics["aep"] >= baseline["aep"] * (1.0 - tolerance)
        and metrics["maximum_fatigue"]
        <= baseline["maximum_fatigue"] * (1.0 + tolerance)
    )


def constrained_score(metrics, baseline):
    aep_ratio = metrics["aep"] / baseline["aep"]
    max_fatigue_ratio = (
        metrics["maximum_fatigue"] / baseline["maximum_fatigue"]
    )
    violation = max(0.0, 1.0 - aep_ratio) + max(
        0.0, max_fatigue_ratio - 1.0
    )
    lcoe_ratio = metrics["lcoe"] / baseline["lcoe"]
    wake_ratio = metrics["wake_loss"] / max(baseline["wake_loss"], 1e-9)
    mean_fatigue_ratio = (
        metrics["mean_fatigue"] / baseline["mean_fatigue"]
    )
    return (
        -1000.0 * violation
        - lcoe_ratio
        - WAKE_LOSS_PENALTY * wake_ratio
        - MEAN_FATIGUE_PENALTY * mean_fatigue_ratio
    )


def is_better(candidate, champion, baseline):
    candidate_feasible = is_feasible(candidate, baseline)
    champion_feasible = is_feasible(champion, baseline)
    if candidate_feasible != champion_feasible:
        return candidate_feasible
    if candidate_feasible:
        return (candidate["lcoe"], candidate["wake_loss"], -candidate["aep"]) < (
            champion["lcoe"],
            champion["wake_loss"],
            -champion["aep"],
        )
    return constrained_score(candidate, baseline) > constrained_score(
        champion, baseline
    )


def adaptive_mutation(generation, stagnant_generations):
    progress = generation / max(GENERATIONS - 1, 1)
    rate = INITIAL_MUTATION_RATE + progress * (
        FINAL_MUTATION_RATE - INITIAL_MUTATION_RATE
    )
    distance = D * (
        INITIAL_MUTATION_DISTANCE_D
        + progress
        * (FINAL_MUTATION_DISTANCE_D - INITIAL_MUTATION_DISTANCE_D)
    )
    if stagnant_generations >= STAGNATION_LIMIT:
        rate = min(0.60, rate * 1.8)
        distance = min(1.5 * D, distance * 1.8)
    return rate, distance


def run_single_optimization(seed, cases, baseline):
    rng = np.random.default_rng(seed)
    speeds, directions, densities, weights = map(
        np.asarray, zip(*cases)
    )
    population = generate_population(POPULATION_SIZE, rng)
    baseline_individual = make_individual(
        repair_layout(centre_in_domain(staggered_layout()), rng)
    )
    population[0] = copy_individual(baseline_individual)
    best = copy_individual(baseline_individual)
    best_metrics = baseline
    stagnant = 0
    history = {"generation": [], "aep": [], "lcoe": [], "maximum_fatigue": []}

    for generation in range(GENERATIONS):
        metrics = [
            evaluate_individual(
                individual, speeds, directions, densities, weights
            )
            for individual in population
        ]
        scores = np.array(
            [constrained_score(item, baseline) for item in metrics]
        )
        ranking = np.argsort(scores)[::-1]
        improved = False
        for index in ranking[: min(FULL_CHECK_COUNT, len(ranking))]:
            if is_better(metrics[index], best_metrics, baseline):
                best = copy_individual(population[index])
                best_metrics = metrics[index]
                improved = True
        stagnant = 0 if improved else stagnant + 1
        history["generation"].append(generation + 1)
        history["aep"].append(best_metrics["aep"])
        history["lcoe"].append(best_metrics["lcoe"])
        history["maximum_fatigue"].append(best_metrics["maximum_fatigue"])
        print(
            f"Seed {seed:>3} | generation {generation + 1:>2}/{GENERATIONS} | "
            f"AEP {best_metrics['aep']:,.1f} MWh | "
            f"LCOE ${best_metrics['lcoe']:.2f}/MWh | "
            f"max fatigue {best_metrics['maximum_fatigue']:.3e}"
        )
        if generation + 1 == GENERATIONS:
            break

        mutation_rate, mutation_distance = adaptive_mutation(
            generation, stagnant
        )
        new_population = [
            copy_individual(population[index])
            for index in ranking[:ELITE_COUNT]
        ]
        immigrant_count = max(1, int(POPULATION_SIZE * 0.05)) #30 Turbines change 1
        """
        new_population.extend(
            make_individual(generate_layout(rng))
            for _ in range(
                min(immigrant_count, POPULATION_SIZE - len(new_population))
            )
        )
        """
        number_of_immigrants = min(
            immigrant_count,
            POPULATION_SIZE - len(new_population),
        )

        for _ in range(number_of_immigrants):
            try:
                new_population.append(
                    make_individual(generate_layout(rng))
                )
            except RuntimeError:
                fallback_parent = population[rng.integers(len(population))]

                new_population.append(
                    mutate_individual(
                       fallback_parent,
                       rng,
                       rate=0.60,
                       distance=1.25 * D,
                    )
                )
        while len(new_population) < POPULATION_SIZE:
            parent_1 = tournament_selection(population, scores, rng)
            parent_2 = tournament_selection(population, scores, rng)
            child = crossover(parent_1, parent_2, rng)
            child["layout"] = repair_layout(child["layout"], rng)
            new_population.append(
                mutate_individual(
                    child, rng, mutation_rate, mutation_distance
                )
            )
        population = new_population
    return {"seed": seed, "individual": best, "search_metrics": best_metrics, "history": history}


def print_metrics(label, metrics):
    print(f"\n===== {label} =====")
    print(f"AEP: {metrics['aep']:,.1f} MWh/year")
    print(f"LCOE: ${metrics['lcoe']:,.2f}/MWh")
    print(f"Wake loss: {metrics['wake_loss']:.2f}%")
    print(f"Maximum fatigue: {metrics['maximum_fatigue']:.3e}")
    print(f"Mean fatigue: {metrics['mean_fatigue']:.3e}")
    print(f"Worst turbine: {metrics['worst_turbine']}")
    print(f"Cable length: {metrics['cable_length_m']:,.1f} m")
    print(f"Road length: {metrics['road_length_m']:,.1f} m")


def make_baseline_graph(layout, metrics, outpath):
    layout = np.asarray(layout) / D
    fatigue = metrics["per_turbine_fatigue"]
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        layout[:, 0], layout[:, 1], c=fatigue, cmap="viridis", s=90
    )
    fig.colorbar(scatter, ax=ax, label="Relative fatigue index")
    ax.set(
        title=(
            "Staggered baseline\n"
            f"AEP {metrics['aep']:,.0f} MWh | "
            f"LCOE ${metrics['lcoe']:.2f}/MWh | "
            f"Max fatigue {metrics['maximum_fatigue']:.2e}"
        ),
        xlabel="X position (D)",
        ylabel="Y position (D)",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def make_progress_graph(history, baseline, outpath):
    generations = history["generation"]
    figure, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for ax, key, label, base in (
        (axes[0], "aep", "AEP (MWh)", baseline["aep"]),
        (axes[1], "lcoe", "LCOE ($/MWh)", baseline["lcoe"]),
        (
            axes[2],
            "maximum_fatigue",
            "Maximum fatigue",
            baseline["maximum_fatigue"],
        ),
    ):
        ax.plot(generations, history[key], color="seagreen")
        ax.axhline(base, color="darkorange", linestyle="--")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Generation")
    figure.suptitle("Best multi-start GA progress vs staggered baseline")
    figure.tight_layout()
    figure.savefig(outpath, dpi=150)
    plt.close(figure)


def make_multi_seed_graph(results, baseline, outpath):
    seeds = [result["seed"] for result in results]
    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, key, label, base in (
        (axes[0], "aep", "AEP (MWh)", baseline["aep"]),
        (axes[1], "lcoe", "LCOE ($/MWh)", baseline["lcoe"]),
        (
            axes[2],
            "maximum_fatigue",
            "Maximum fatigue",
            baseline["maximum_fatigue"],
        ),
    ):
        ax.plot(
            seeds,
            [result["metrics"][key] for result in results],
            marker="o",
        )
        ax.axhline(base, color="darkorange", linestyle="--")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Seed")
    figure.suptitle("Full-dataset results for every GA seed")
    figure.tight_layout()
    figure.savefig(outpath, dpi=150)
    plt.close(figure)

def dominant_yaw_snapshot(individual, speeds, directions):
    """Return the applied yaw of every turbine for the dominant wind sector."""
    speeds = np.asarray(speeds, dtype=float)
    directions = np.asarray(directions, dtype=float) % 360.0

    sector_width = 360.0 / DIRECTION_BINS
    sector_indices = np.floor(
        directions / sector_width
    ).astype(int)

    # Select the wind-direction sector containing the most weather samples.
    dominant_sector = int(
        np.argmax(
            np.bincount(
                sector_indices,
                minlength=DIRECTION_BINS,
            )
        )
    )

    selected = sector_indices == dominant_sector

    # Use the centre of the dominant direction sector.
    dominant_direction = (
        dominant_sector + 0.5
    ) * sector_width

    # Convert the representative wind speed from 100 m to 95 m hub height.
    representative_speed = float(
        wind_speed_at_hub(speeds[selected]).mean()
    )

    # Final yaw selected by the winning GA individual for this sector.
    sector_yaw = float(
        individual["yaw"][dominant_sector]
    )

    rotated_layout = rotate_layout_to_wind_frame(
        individual["layout"],
        dominant_direction,
    )

    # Apply the sector yaw only to turbines that have a downstream wake target.
    turbine_yaws = _active_yaw_angles(
        rotated_layout,
        representative_speed,
        sector_yaw,
    )

    return (
        dominant_direction,
        representative_speed,
        turbine_yaws,
    )


def make_best_layout_graph(
    baseline_layout,
    baseline,
    best,
    speeds,
    directions,
    outpath,
):
    (
        dominant_direction,
        representative_speed,
        best_yaws,
    ) = dominant_yaw_snapshot(
        best["individual"],
        speeds,
        directions,
    )

    layouts = [
        np.asarray(baseline_layout, dtype=float) / D,
        np.asarray(
            best["individual"]["layout"],
            dtype=float,
        ) / D,
    ]

    metrics = [
        baseline,
        best["metrics"],
    ]

    titles = [
        "Staggered baseline",
        f"Best seed {best['seed']}",
    ]

    # Baseline turbines have no yaw steering.
    yaw_values = [
        np.zeros(TURBINES),
        best_yaws,
    ]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, 7),
        sharex=True,
        sharey=True,
    )

    for panel, (
        ax,
        layout,
        item,
        title,
        turbine_yaws,
    ) in enumerate(
        zip(
            axes,
            layouts,
            metrics,
            titles,
            yaw_values,
        )
    ):
        scatter = ax.scatter(
            layout[:, 0],
            layout[:, 1],
            c=item["per_turbine_fatigue"],
            cmap="viridis",
            s=100,
            zorder=2,
        )

        # The model treats dominant_direction as the downstream flow axis.
        # Turbines face in the opposite direction, with yaw added.
        facing_directions = (
            dominant_direction
            + 180.0
            + turbine_yaws
        ) % 360.0

        facing_radians = np.radians(facing_directions)

        # Arrow length is measured in rotor-diameter units.
        arrow_length = 0.90
        arrow_x = arrow_length * np.cos(facing_radians)
        arrow_y = arrow_length * np.sin(facing_radians)

        ax.quiver(
            layout[:, 0],
            layout[:, 1],
            arrow_x,
            arrow_y,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="black",
            width=0.006,
            headwidth=4,
            headlength=5,
            headaxislength=4.5,
            pivot="middle",
            zorder=3,
        )

        for turbine_number, ((x, y), yaw) in enumerate(
            zip(layout, turbine_yaws),
            start=1,
        ):
            if panel == 0:
                label = f"T{turbine_number}"
            else:
                label = (
                    f"T{turbine_number}\n"
                    f"{yaw:+.1f}°"
                )

            ax.annotate(
                label,
                (x, y),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=7,
                zorder=4,
            )

        # Draw a larger blue arrow showing downstream wind flow.
        flow_angle = np.radians(dominant_direction)
        flow_dx = 0.12 * np.cos(flow_angle)
        flow_dy = 0.12 * np.sin(flow_angle)

        flow_centre_x = 0.14
        flow_centre_y = 0.88

        ax.annotate(
            "",
            xy=(
                flow_centre_x + flow_dx / 2.0,
                flow_centre_y + flow_dy / 2.0,
            ),
            xytext=(
                flow_centre_x - flow_dx / 2.0,
                flow_centre_y - flow_dy / 2.0,
            ),
            xycoords="axes fraction",
            arrowprops={
                "arrowstyle": "->",
                "color": "dodgerblue",
                "linewidth": 2.5,
            },
        )

        ax.text(
            0.03,
            0.97,
            "Wind flow",
            transform=ax.transAxes,
            color="dodgerblue",
            fontsize=8,
            va="top",
        )

        ax.set_title(
            f"{title}\n"
            f"{item['aep']:,.0f} MWh | "
            f"${item['lcoe']:.2f}/MWh"
        )
        ax.set_xlabel("X position (D)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)

        figure.colorbar(
            scatter,
            ax=ax,
            label="Relative fatigue index",
        )

    axes[0].set_ylabel("Y position (D)")

    figure.suptitle(
        "Final GA layout and turbine facing directions\n"
        f"Dominant sector: {dominant_direction:.1f}° | "
        f"Representative hub-height speed: "
        f"{representative_speed:.1f} m/s\n"
        "Black arrows = turbine facing direction | "
        "Blue arrow = wind flow"
    )

    figure.tight_layout()
    figure.savefig(
        outpath,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def print_settings():
    print("\n===== EDITABLE EXPERIMENT SETTINGS =====")
    print(f"Seeds: {SEEDS}")
    print(f"Population: {POPULATION_SIZE} | Generations: {GENERATIONS}")
    print(
        f"Domain: {SEARCH_WIDTH_D}D x {SEARCH_HEIGHT_D}D | "
        f"Minimum spacing: {MIN_SPACING_D}D"
    )
    print(
        f"Wind rose: {DIRECTION_BINS} direction x {SPEED_BINS} speed bins"
    )


def run_multi_start():
    print_settings()
    print("\nLoading synchronized wind, temperature and pressure data...")
    weather = load_real_weather_data()
    speeds = weather["wind_speed_m_s"].to_numpy()
    directions = weather["wind_direction_deg"].to_numpy()
    densities = weather["air_density_kg_m3"].to_numpy()
    print(f"Loaded {len(weather)} synchronized samples.")

    baseline_layout = centre_in_domain(staggered_layout())
    baseline_individual = make_individual(baseline_layout)
    full_weights = np.ones(len(weather))
    baseline = evaluate_individual(
        baseline_individual, speeds, directions, densities, full_weights
    )
    print_metrics("STAGGERED BASELINE", baseline)
    if SAVE_GRAPHS:
        make_baseline_graph(
            baseline_layout,
            baseline,
            os.path.join(SCRIPT_DIR, "staggered_baseline.png"),
        )

    cases = build_wind_rose_cases(
        speeds,
        directions,
        densities,
        DIRECTION_BINS,
        SPEED_BINS,
    )
    case_speeds, case_directions, case_densities, case_weights = map(
        np.asarray, zip(*cases)
    )
    search_baseline = evaluate_individual(
        baseline_individual,
        case_speeds,
        case_directions,
        case_densities,
        case_weights,
    )

    results = []
    for run_number, seed in enumerate(SEEDS, start=1):
        print(f"\n===== GA START {run_number}/{len(SEEDS)} | SEED {seed} =====")
        result = run_single_optimization(seed, cases, search_baseline)
        result["metrics"] = evaluate_individual(
            result["individual"],
            speeds,
            directions,
            densities,
            full_weights,
        )
        result["constraints_satisfied"] = is_feasible(
            result["metrics"], baseline
        )
        results.append(result)
        print_metrics(f"SEED {seed} FULL-DATASET WINNER", result["metrics"])
        print(f"Constraints satisfied: {result['constraints_satisfied']}")

    feasible = [result for result in results if result["constraints_satisfied"]]
    if not feasible:
        raise RuntimeError(
            "No GA seed satisfied both baseline constraints on the full dataset."
        )
    best = min(
        feasible,
        key=lambda result: (
            result["metrics"]["lcoe"],
            result["metrics"]["wake_loss"],
            -result["metrics"]["aep"],
        ),
    )
    print_metrics(f"FINAL WINNER - SEED {best['seed']}", best["metrics"])
    print(
        f"AEP change: {(best['metrics']['aep'] / baseline['aep'] - 1) * 100:+.3f}%"
    )
    print(
        f"LCOE change: {(best['metrics']['lcoe'] / baseline['lcoe'] - 1) * 100:+.3f}%"
    )
    print(
        "Maximum-fatigue change: "
        f"{(best['metrics']['maximum_fatigue'] / baseline['maximum_fatigue'] - 1) * 100:+.3f}%"
    )

    if SAVE_GRAPHS:
        make_progress_graph(
            best["history"],
            search_baseline,
            os.path.join(SCRIPT_DIR, "ga_progress.png"),
        )
        make_multi_seed_graph(
            results,
            baseline,
            os.path.join(SCRIPT_DIR, "ga_multi_seed_results.png"),
        )
        make_best_layout_graph(
            baseline_layout,
            baseline,
            best,
            speeds,
            directions,
            os.path.join(SCRIPT_DIR, "ga_best_layout.png"),
        )
        print(
            "\nGraphs saved: staggered_baseline.png, ga_progress.png, "
            "ga_multi_seed_results.png, ga_best_layout.png"
        )
    return {"baseline_metrics": baseline, "best": best, "runs": results}


def run_optimization(seed=None, make_graphs=True):
    """Compatibility entrypoint; edit SEEDS or pass one seed for a peer run."""
    global SEEDS, SAVE_GRAPHS
    previous_seeds, previous_graphs = SEEDS, SAVE_GRAPHS
    if seed is not None:
        SEEDS = [seed]
    SAVE_GRAPHS = make_graphs
    try:
        return run_multi_start()
    finally:
        SEEDS, SAVE_GRAPHS = previous_seeds, previous_graphs


if __name__ == "__main__":
    run_multi_start()
