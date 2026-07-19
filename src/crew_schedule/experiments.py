"""Reproducible random instances and cross-solver benchmark tables."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from .annealing import solve_annealing
from .classical import cpm_analysis, solve_cpm_list
from .models import ProjectInstance, Task
from .qubo import build_qubo


def generate_random_project(
    task_count: int,
    crew_limit: int,
    *,
    seed: int,
    extra_edge_probability: float = 0.15,
) -> ProjectInstance:
    """Generate a connected random DAG in an already topological task order."""

    if task_count < 2:
        raise ValueError("task_count must be at least two")
    rng = np.random.default_rng(seed)
    tasks = tuple(
        Task(f"t{index}", f"Task {index + 1}", int(rng.integers(1, 5)))
        for index in range(task_count)
    )
    edges: set[tuple[str, str]] = set()
    for successor in range(1, task_count):
        required_predecessor = int(rng.integers(0, successor))
        edges.add((f"t{required_predecessor}", f"t{successor}"))
        for predecessor in range(successor):
            if predecessor != required_predecessor and rng.random() < extra_edge_probability:
                edges.add((f"t{predecessor}", f"t{successor}"))
    return ProjectInstance(
        tasks,
        tuple(sorted(edges)),
        crew_limit=crew_limit,
        name=f"random_n{task_count}_r{crew_limit}_s{seed}",
    )


def run_benchmarks(
    *,
    task_counts: Iterable[int] = (4, 6, 8, 10, 12),
    crew_limits: Iterable[int] = (2, 3),
    instances_per_cell: int = 2,
    repetitions: int = 3,
    base_seed: int = 20_260_719,
    annealing_params: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Run the CPM-list and annealing comparison on a deterministic matrix."""

    params = {
        "num_reads": 50,
        "num_sweeps": 500,
        "beta_schedule_type": "geometric",
    }
    if annealing_params:
        params.update(annealing_params)

    rows: list[dict[str, object]] = []
    cell_index = 0
    for task_count in task_counts:
        for crew_limit in crew_limits:
            for instance_index in range(instances_per_cell):
                instance_seed = base_seed + 1_000 * cell_index + instance_index
                instance = generate_random_project(
                    task_count, crew_limit, seed=instance_seed
                )
                classical = solve_cpm_list(instance)
                cpm = cpm_analysis(instance)
                bqm, _ = build_qubo(instance)
                lower_bound = int(cpm["combined_lower_bound"])
                classical_makespan = classical.schedule.makespan
                for repetition in range(repetitions):
                    anneal_seed = base_seed + 100_000 + cell_index * 100 + repetition
                    annealed = solve_annealing(instance, seed=anneal_seed, **params)
                    makespan = annealed.schedule.makespan if annealed.schedule else np.nan
                    rows.append(
                        {
                            "instance": instance.name,
                            "task_count": task_count,
                            "crew_limit": crew_limit,
                            "instance_index": instance_index,
                            "repetition": repetition,
                            "instance_seed": instance_seed,
                            "anneal_seed": anneal_seed,
                            "lower_bound": lower_bound,
                            "classical_makespan": classical_makespan,
                            "classical_runtime": classical.runtime,
                            "annealing_feasible": annealed.feasible,
                            "annealing_makespan": makespan,
                            "annealing_runtime": annealed.runtime,
                            "annealing_build_time": annealed.metadata["build_time"],
                            "annealing_sample_time": annealed.metadata["sample_time"],
                            "feasible_read_rate": annealed.metadata["feasible_read_rate"],
                            "gap_to_lower_bound": (
                                (makespan - lower_bound) / lower_bound
                                if annealed.feasible
                                else np.nan
                            ),
                            "improvement_vs_classical": (
                                classical_makespan - makespan
                                if annealed.feasible
                                else np.nan
                            ),
                            "bqm_variables": bqm.num_variables,
                            "bqm_interactions": bqm.num_interactions,
                        }
                    )
                cell_index += 1
    return pd.DataFrame(rows)


def consistency_summary(benchmark: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated annealing runs into per-instance consistency data."""

    return (
        benchmark.groupby(
            ["instance", "task_count", "crew_limit"], as_index=False
        )
        .agg(
            feasible_run_rate=("annealing_feasible", "mean"),
            median_makespan=("annealing_makespan", "median"),
            makespan_std=("annealing_makespan", "std"),
            median_runtime=("annealing_runtime", "median"),
            median_feasible_read_rate=("feasible_read_rate", "median"),
        )
        .fillna({"makespan_std": 0.0})
    )

