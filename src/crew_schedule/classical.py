"""Critical-path analysis and deterministic resource-feasible list scheduling."""

from __future__ import annotations

from math import ceil
from time import perf_counter

import networkx as nx

from .models import ProjectInstance, Schedule, SolverResult, validate_schedule


def cpm_analysis(instance: ProjectInstance) -> dict[str, object]:
    """Return precedence-only CPM values and a simple resource lower bound."""

    graph = instance.graph
    durations = {task.id: task.duration for task in instance.tasks}
    order = list(nx.topological_sort(graph))
    earliest_start: dict[str, int] = {}
    earliest_finish: dict[str, int] = {}
    for task_id in order:
        earliest_start[task_id] = max(
            (earliest_finish[p] for p in graph.predecessors(task_id)), default=0
        )
        earliest_finish[task_id] = earliest_start[task_id] + durations[task_id]

    critical_path_length = max(earliest_finish.values())
    latest_finish: dict[str, int] = {}
    latest_start: dict[str, int] = {}
    for task_id in reversed(order):
        latest_finish[task_id] = min(
            (latest_start[s] for s in graph.successors(task_id)),
            default=critical_path_length,
        )
        latest_start[task_id] = latest_finish[task_id] - durations[task_id]

    total_float = {
        task_id: latest_start[task_id] - earliest_start[task_id] for task_id in order
    }
    critical_tasks = [task_id for task_id in order if total_float[task_id] == 0]
    resource_bound = ceil(sum(durations.values()) / instance.crew_limit)
    return {
        "earliest_start": earliest_start,
        "earliest_finish": earliest_finish,
        "latest_start": latest_start,
        "latest_finish": latest_finish,
        "total_float": total_float,
        "critical_tasks": critical_tasks,
        "critical_path_length": critical_path_length,
        "resource_lower_bound": resource_bound,
        "combined_lower_bound": max(critical_path_length, resource_bound),
    }


def _bottom_levels(instance: ProjectInstance) -> dict[str, int]:
    graph = instance.graph
    durations = {task.id: task.duration for task in instance.tasks}
    levels: dict[str, int] = {}
    for task_id in reversed(list(nx.topological_sort(graph))):
        levels[task_id] = durations[task_id] + max(
            (levels[s] for s in graph.successors(task_id)), default=0
        )
    return levels


def solve_cpm_list(instance: ProjectInstance) -> SolverResult:
    """Schedule ready work using descending remaining critical-path priority."""

    started_at = perf_counter()
    graph = instance.graph
    durations = {task.id: task.duration for task in instance.tasks}
    priorities = _bottom_levels(instance)
    unscheduled = set(durations)
    completed: set[str] = set()
    active: dict[str, int] = {}
    starts: dict[str, int] = {}
    now = 0

    while unscheduled or active:
        finished = sorted(task_id for task_id, end in active.items() if end <= now)
        for task_id in finished:
            completed.add(task_id)
            del active[task_id]

        ready = [
            task_id
            for task_id in unscheduled
            if set(graph.predecessors(task_id)).issubset(completed)
        ]
        ready.sort(key=lambda task_id: (-priorities[task_id], task_id))
        for task_id in ready[: instance.crew_limit - len(active)]:
            starts[task_id] = now
            active[task_id] = now + durations[task_id]
            unscheduled.remove(task_id)

        if active:
            now = min(active.values())
        elif unscheduled:
            raise RuntimeError("no schedulable tasks remain; graph validation should prevent this")

    makespan = max(starts[task.id] + task.duration for task in instance.tasks)
    schedule = Schedule(starts, makespan)
    feasible, errors = validate_schedule(instance, schedule)
    cpm = cpm_analysis(instance)
    return SolverResult(
        method="CPM-priority list scheduling",
        schedule=schedule,
        feasible=feasible,
        energy=None,
        runtime=perf_counter() - started_at,
        metadata={"validation_errors": errors, "priorities": priorities, **cpm},
    )

