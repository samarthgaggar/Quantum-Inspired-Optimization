"""D-Wave classical simulated annealing and compact parameter tuning."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from time import perf_counter
from typing import Iterable, Sequence

import pandas as pd
from dwave.samplers import SimulatedAnnealingSampler

from .models import ProjectInstance, SolverResult
from .qubo import QuboEncoding, build_qubo, decode_sample


@dataclass(frozen=True)
class AnnealingConfig:
    name: str
    num_reads: int
    num_sweeps: int
    beta_schedule_type: str = "geometric"
    beta_range: tuple[float, float] | None = None


def _sample_bqm(
    instance: ProjectInstance,
    bqm,
    encoding: QuboEncoding,
    *,
    num_reads: int,
    num_sweeps: int,
    beta_range: Sequence[float] | None,
    beta_schedule_type: str,
    seed: int,
    build_time: float,
) -> SolverResult:
    sampler = SimulatedAnnealingSampler()
    sampled_at = perf_counter()
    sampleset = sampler.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        beta_range=beta_range,
        beta_schedule_type=beta_schedule_type,
        seed=seed,
    )
    sample_time = perf_counter() - sampled_at

    feasible_reads = 0
    best_schedule = None
    best_energy = None
    best_finish = None
    for datum in sampleset.data(fields=["sample", "energy", "num_occurrences"], sorted_by="energy"):
        schedule, _ = decode_sample(datum.sample, encoding)
        if schedule is None:
            continue
        feasible_reads += int(datum.num_occurrences)
        if best_schedule is None:
            best_schedule = schedule
            best_energy = float(datum.energy)
            selected_finish = [
                time
                for time in encoding.domains[encoding.finish_id]
                if datum.sample[encoding.start_variables[(encoding.finish_id, time)]] == 1
            ]
            best_finish = selected_finish[0] if selected_finish else None

    info = dict(sampleset.info)
    return SolverResult(
        method="Simulated annealing",
        schedule=best_schedule,
        feasible=best_schedule is not None,
        energy=best_energy,
        runtime=build_time + sample_time,
        metadata={
            "build_time": build_time,
            "sample_time": sample_time,
            "num_reads": num_reads,
            "num_sweeps": num_sweeps,
            "beta_range": info.get("beta_range", beta_range),
            "beta_schedule_type": beta_schedule_type,
            "seed": seed,
            "feasible_reads": feasible_reads,
            "feasible_read_rate": feasible_reads / num_reads,
            "variables": bqm.num_variables,
            "interactions": bqm.num_interactions,
            "selected_finish": best_finish,
            "sampler_timing": info.get("timing", {}),
        },
    )


def solve_annealing(
    instance: ProjectInstance,
    *,
    num_reads: int = 200,
    num_sweeps: int = 2_000,
    beta_range: Sequence[float] | None = None,
    beta_schedule_type: str = "geometric",
    seed: int = 1234,
) -> SolverResult:
    """Build and sample a scheduling QUBO, returning the best feasible read."""

    build_started = perf_counter()
    bqm, encoding = build_qubo(instance)
    build_time = perf_counter() - build_started
    return _sample_bqm(
        instance,
        bqm,
        encoding,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        beta_range=beta_range,
        beta_schedule_type=beta_schedule_type,
        seed=seed,
        build_time=build_time,
    )


def tune_annealing(
    instance: ProjectInstance,
    *,
    seeds: Iterable[int] = (101, 202, 303),
) -> tuple[AnnealingConfig, pd.DataFrame]:
    """Evaluate four compact settings and select by feasibility, quality, time."""

    probe = solve_annealing(instance, num_reads=1, num_sweeps=10, seed=17)
    automatic = probe.metadata.get("beta_range")
    widened: tuple[float, float] | None = None
    if automatic is not None:
        widened = (float(automatic[0]) * 0.5, float(automatic[1]) * 2.0)

    configs = (
        AnnealingConfig("quick", 50, 500, "geometric", None),
        AnnealingConfig("more_reads", 200, 500, "geometric", None),
        AnnealingConfig("more_sweeps", 50, 2_000, "geometric", None),
        AnnealingConfig("linear_wide", 200, 2_000, "linear", widened),
    )
    rows: list[dict[str, object]] = []
    for config in configs:
        for seed in seeds:
            result = solve_annealing(
                instance,
                num_reads=config.num_reads,
                num_sweeps=config.num_sweeps,
                beta_range=config.beta_range,
                beta_schedule_type=config.beta_schedule_type,
                seed=seed,
            )
            rows.append(
                {
                    "config": config.name,
                    "seed": seed,
                    "feasible": result.feasible,
                    "makespan": result.schedule.makespan if result.schedule else None,
                    "runtime": result.runtime,
                    "feasible_read_rate": result.metadata["feasible_read_rate"],
                    "num_reads": config.num_reads,
                    "num_sweeps": config.num_sweeps,
                    "schedule_type": config.beta_schedule_type,
                    "beta_range": str(config.beta_range or "automatic"),
                }
            )
    frame = pd.DataFrame(rows)
    ranking: list[tuple[tuple[float, float, float], AnnealingConfig]] = []
    for config in configs:
        subset = frame[frame["config"] == config.name]
        success_rate = float(subset["feasible"].mean())
        feasible_makespans = subset.loc[subset["feasible"], "makespan"].astype(float)
        median_makespan = median(feasible_makespans) if len(feasible_makespans) else float("inf")
        median_runtime = median(subset["runtime"].astype(float))
        ranking.append(((-success_rate, median_makespan, median_runtime), config))
    ranking.sort(key=lambda item: item[0])
    return ranking[0][1], frame

