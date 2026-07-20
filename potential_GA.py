import os
 
import matplotlib.pyplot as plt
import numpy as np
 
from layouts import grid_layout, staggered_layout
from main import calculate_aep, load_real_wind_data
from turbine import vestas
 
 
turbines = 25
D = vestas.rotor_diameter
 
width = 20 * D
height = 22.5 * D
min_spacing = 4 * D
 
 
# LAYOUTS
 
def sort_layout(layout):
    order = np.lexsort((layout[:, 1], layout[:, 0]))
    return layout[order]
 
 
def generate_layout(rng):
    # Generate one valid random layout
    while True:
        layout = []
 
        for _ in range(10000):
            candidate = np.array([
                rng.uniform(0, width),
                rng.uniform(0, height),
            ])
 
            valid_position = all(
                np.linalg.norm(candidate - existing) >= min_spacing
                for existing in layout
            )
 
            if valid_position:
                layout.append(candidate)
 
            if len(layout) == turbines:
                return sort_layout(np.array(layout))
 
 
def generate_population(size, rng, starting_layout=None):
    if starting_layout is None:
        return [generate_layout(rng) for _ in range(size)]
 
    population = [sort_layout(np.array(starting_layout, dtype=float))]
    variant_count = max(1, int(size * 0.7))
 
    while len(population) < variant_count:
        variant = mutate(
            population[0],
            rng,
            mutation_rate=0.35,
            mutation_distance=0.35 * D,
        )
        population.append(repair_layout(variant, rng))
 
    # The remaining slots are fully random layouts. This matters more than it
    # looks: without a steady stream of genuinely unrelated genotypes, the
    # population converges around the staggered seed within a few
    # generations and crossover just recombines near-identical parents.
    while len(population) < size:
        population.append(generate_layout(rng))
 
    return population
 
 
# SELECTION
 
def selection(population, fitness_scores, rng, tournament_size=3):
    competitors = rng.choice(
        len(population),
        size=tournament_size,
        replace=False,
    )
 
    winner = max(
        competitors,
        key=lambda index: fitness_scores[index],
    )
 
    return population[winner].copy()
 
 
# CROSSOVER
 
def crossover(parent1, parent2, rng):
    # Inherit complete turbine positions. Blending coordinates can pull two
    # otherwise well-spaced turbines into the same area and forces the repair
    # function to replace much of the child with random positions.
    inherit_from_parent1 = rng.random(turbines) < 0.5
    child = parent2.copy()
    child[inherit_from_parent1] = parent1[inherit_from_parent1]
    return child
 
 
# MUTATION
 
def mutate(
    layout,
    rng,
    mutation_rate=0.10,
    mutation_distance=0.5 * D,
):
    mutated_layout = layout.copy()
 
    for i in range(turbines):
        if rng.random() < mutation_rate:
            mutated_layout[i] += rng.normal(0, mutation_distance, size=2)
 
    mutated_layout[:, 0] = np.clip(mutated_layout[:, 0], 0, width)
    mutated_layout[:, 1] = np.clip(mutated_layout[:, 1], 0, height)
 
    return mutated_layout
 
 
# REPAIR
 
def repair_layout(layout, rng):
    repaired_layout = []
 
    for position in layout:
        candidate = position.copy()
        candidate[0] = np.clip(candidate[0], 0, width)
        candidate[1] = np.clip(candidate[1], 0, height)
 
        attempts = 0
 
        while any(
            np.linalg.norm(candidate - existing) < min_spacing
            for existing in repaired_layout
        ):
            attempts += 1
 
            # First try a nearby position so useful parent geometry is kept.
            # Use a random position only if local repair repeatedly fails.
            if attempts <= 100:
                candidate = position + rng.normal(0, 0.35 * D, size=2)
                candidate[0] = np.clip(candidate[0], 0, width)
                candidate[1] = np.clip(candidate[1], 0, height)
            else:
                candidate = np.array([
                    rng.uniform(0, width),
                    rng.uniform(0, height),
                ])
 
            if attempts == 5000:
                return generate_layout(rng)
 
        repaired_layout.append(candidate)
 
    return sort_layout(np.array(repaired_layout))
 
 
# GENETIC ALGORITHM
 
def genetic_algorithm(
    fitness_function,
    full_fitness_function,
    starting_layout=None,
    population_size=24,
    generations=20,
    elite_count=2,
    mutation_rate=0.20,
    stagnation_limit=3,
    full_check_count=3,
    seed=42,
):
    """Evolve turbine layouts starting from `starting_layout`.
 
    `fitness_function` scores the population each generation and can be a
    fast subsample-based estimate — it only needs to rank layouts relative
    to each other, so approximation is fine here.
 
    `full_fitness_function` is the true full-dataset AEP. The staggered
    baseline is evaluated once, then the best `full_check_count` layouts
    from every generation are checked using all wind data. This prevents
    approximation error in the 1,000-hour search sample from hiding a
    genuinely better layout.
 
    Returns (best_layout, best_aep, starting_aep, history), where best_aep
    and starting_aep are both full-dataset AEP values.
    """
    rng = np.random.default_rng(seed)
    population = generate_population(population_size, rng, starting_layout)
 
    starting_aep = (
        full_fitness_function(starting_layout)
        if starting_layout is not None
        else None
    )
 
    history = {
        "generation": [],
        "best_aep": [],
        "difference": [],
        "generation_search_score": [],
    }

    # The staggered baseline is already a valid champion. Starting from it
    # prevents an inferior random layout from being reported as the best.
    if starting_layout is not None:
        best_layout = sort_layout(np.array(starting_layout, dtype=float))
        best_aep = starting_aep
    else:
        best_layout = None
        best_aep = -np.inf

    stagnant_generations = 0
 
    for generation in range(1, generations + 1):
        search_scores = np.array([
            fitness_function(layout)
            for layout in population
        ])
        ranking = np.argsort(search_scores)[::-1]
 
        generation_search_score = search_scores[ranking[0]]

        # Fully check the best few layouts from EVERY generation. The old
        # version checked a layout only when it beat the all-time approximate
        # score, which could permanently ignore a genuinely better full-year
        # layout because the 1,000-hour search score is only an estimate.
        improved = False
        candidates_to_check = ranking[:min(full_check_count, len(ranking))]

        for index in candidates_to_check:
            candidate_layout = population[index]
            candidate_aep = full_fitness_function(candidate_layout)

            if candidate_aep > best_aep:
                best_aep = candidate_aep
                best_layout = candidate_layout.copy()
                improved = True

        if improved:
            stagnant_generations = 0
        else:
            stagnant_generations += 1
 
        difference = (
            best_aep - starting_aep if starting_aep is not None else 0.0
        )
 
        history["generation"].append(generation)
        history["best_aep"].append(best_aep)
        history["difference"].append(difference)
        history["generation_search_score"].append(generation_search_score)
 
        if starting_aep is not None:
            pct = (difference / starting_aep) * 100
            print(
                f"Generation {generation:>2}/{generations} | "
                f"Estimated best: {generation_search_score:,.1f} MWh | "
                f"AEP: {best_aep:,.1f} MWh | "
                f"AEP Base: {starting_aep:,.1f} MWh | "
                f"Difference: {difference:+,.1f} MWh ({pct:+.3f}%)"
            )
        else:
            print(
                f"Generation {generation:>2}/{generations} | "
                f"AEP: {best_aep:,.1f} MWh"
            )
 
        if generation == generations:
            break
 
        # Adaptive mutation: if the population hasn't produced an
        # improvement for a few generations in a row, temporarily widen the
        # search (higher mutation rate + larger step size) to escape the
        # plateau, then relax back once progress resumes.
        if stagnant_generations >= stagnation_limit:
            active_mutation_rate = min(0.6, mutation_rate * 2)
            active_mutation_distance = 1.0 * D
        else:
            active_mutation_rate = mutation_rate
            active_mutation_distance = 0.5 * D
 
        new_population = [
            population[index].copy()
            for index in ranking[:elite_count]
        ]
 
        # Random immigrants: keep feeding in fully fresh, unrelated layouts
        # every generation so the population never fully converges around
        # a single genotype the way crossover-only reproduction tends to.
        immigrant_count = max(1, int(population_size * 0.15))
        for _ in range(immigrant_count):
            if len(new_population) < population_size:
                new_population.append(generate_layout(rng))
 
        while len(new_population) < population_size:
            parent1 = selection(population, search_scores, rng)
            parent2 = selection(population, search_scores, rng)
 
            child = crossover(parent1, parent2, rng)
            child = mutate(
                child,
                rng,
                active_mutation_rate,
                active_mutation_distance,
            )
            child = repair_layout(child, rng)
 
            new_population.append(child)
 
        population = new_population
 
    return best_layout, best_aep, starting_aep, history
 
 
# GRAPHS
 
def make_progress_graph(history, outpath):
    generations = history["generation"]
    difference = history["difference"]
 
    plt.figure(figsize=(8, 5))
    plt.plot(
        generations,
        difference,
        marker="o",
        color="seagreen",
        linewidth=2,
    )
    plt.axhline(0, color="darkorange", linestyle="--", label="Staggered baseline")
 
    plt.xticks(generations)
    plt.xlabel("Generation")
    plt.ylabel("AEP gain over staggered baseline (MWh)")
    plt.title("Genetic Algorithm Progress vs. Staggered Baseline")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
 
 
def make_best_layout_graph(staggered, best_layout, best_aep, starting_aep, outpath):
    staggered = np.array(staggered) / D
    best = np.array(best_layout) / D
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
 
    axes[0].scatter(staggered[:, 0], staggered[:, 1], color="darkorange", s=45)
    axes[0].set_title(f"Staggered baseline\n{starting_aep:,.1f} MWh")
 
    axes[1].scatter(best[:, 0], best[:, 1], color="seagreen", s=45)
    diff = best_aep - starting_aep
    axes[1].set_title(f"Best GA layout\n{best_aep:,.1f} MWh (+{diff:,.1f} MWh)")
 
    for ax in axes:
        ax.set_xlim(-1, width / D + 1)
        ax.set_ylim(-1, height / D + 1)
        ax.set_xlabel("X position (D)")
        ax.grid(alpha=0.25)
 
    axes[0].set_ylabel("Y position (D)")
 
    plt.suptitle("Best Genetic Algorithm Layout vs. Staggered Baseline")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
 
 
# RUN EXPERIMENT
 
def run_optimization():
    print("Loading wind data...")
    speeds, directions, samples = load_real_wind_data()
    print(f"Loaded {samples} wind samples.")
 
    staggered_positions = staggered_layout()
 
    # `fitness` (fast, subsample-based) ranks the population each
    # generation. `full_fitness` (the true full-dataset AEP) checks the
    # baseline and the best few candidates from every generation.
    rng = np.random.default_rng(42)
    sample_size = min(1000, len(speeds))
    sample_indices = rng.choice(len(speeds), sample_size, replace=False)
    search_speeds = speeds[sample_indices]
    search_directions = directions[sample_indices]
 
    def fitness(layout):
        aep, _ = calculate_aep(layout, search_speeds, search_directions)
        return aep
 
    def full_fitness(layout):
        aep, _ = calculate_aep(layout, speeds, directions)
        return aep
 
    print("\nRunning genetic algorithm...\n")
 
    best_layout, best_aep, staggered_aep, history = genetic_algorithm(
        fitness_function=fitness,
        full_fitness_function=full_fitness,
        starting_layout=np.array(staggered_positions),
        population_size=50,
        generations=30,
        elite_count=3,
        mutation_rate=0.25,
        full_check_count=2,
        seed=42,
    )
 
    _, staggered_wake_loss = calculate_aep(staggered_positions, speeds, directions)
    _, best_wake_loss = calculate_aep(best_layout, speeds, directions)
    final_diff = best_aep - staggered_aep
    final_pct = (final_diff / staggered_aep) * 100
 
    print("\n===== FINAL RESULT =====")
    print(f"Staggered baseline: {staggered_aep:,.1f} MWh | Wake loss: {staggered_wake_loss:.2f}%")
    print(
        f"Best GA layout:     {best_aep:,.1f} MWh | Wake loss: {best_wake_loss:.2f}% | "
        f"Difference: +{final_diff:,.1f} MWh ({final_pct:+.3f}%)"
    )
 
    script_dir = os.path.dirname(os.path.abspath(__file__))
 
    make_progress_graph(
        history,
        os.path.join(script_dir, "ga_progress.png"),
    )
 
    make_best_layout_graph(
        staggered_positions,
        best_layout,
        best_aep,
        staggered_aep,
        os.path.join(script_dir, "ga_best_layout.png"),
    )
 
    print("\nGraphs saved:")
    print("ga_progress.png")
    print("ga_best_layout.png")
 
    return {
        "staggered_aep": staggered_aep,
        "best_aep": best_aep,
        "best_layout": best_layout,
        "history": history,
    }
 
 
if __name__ == "__main__":
    results = run_optimization()