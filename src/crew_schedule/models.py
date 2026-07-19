"""Core data models and independent schedule validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Any, Mapping

import networkx as nx


@dataclass(frozen=True, slots=True)
class Task:
    """A non-preemptive task that occupies one crew for ``duration`` slots."""

    id: str
    name: str
    duration: int

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("task id must be a non-empty string")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"task {self.id!r} must have a non-empty name")
        if isinstance(self.duration, bool) or not isinstance(self.duration, int) or self.duration <= 0:
            raise ValueError(f"task {self.id!r} duration must be a positive integer")


@dataclass(frozen=True)
class ProjectInstance:
    """A single-mode resource-constrained project scheduling instance."""

    tasks: tuple[Task, ...]
    precedence: tuple[tuple[str, str], ...]
    crew_limit: int
    horizon: int | None = None
    name: str = "project"

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "precedence", tuple(tuple(edge) for edge in self.precedence))
        if not self.tasks:
            raise ValueError("a project must contain at least one task")
        if isinstance(self.crew_limit, bool) or not isinstance(self.crew_limit, int) or self.crew_limit <= 0:
            raise ValueError("crew_limit must be a positive integer")

        ids = [task.id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task ids must be unique")
        known = set(ids)
        for edge in self.precedence:
            if len(edge) != 2:
                raise ValueError(f"invalid precedence edge {edge!r}")
            predecessor, successor = edge
            if predecessor not in known or successor not in known:
                raise ValueError(f"precedence edge {edge!r} references an unknown task")
            if predecessor == successor:
                raise ValueError("self precedence is not allowed")

        graph = nx.DiGraph()
        graph.add_nodes_from(ids)
        graph.add_edges_from(self.precedence)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("precedence graph must be acyclic")

        safe_horizon = sum(task.duration for task in self.tasks)
        resolved_horizon = safe_horizon if self.horizon is None else self.horizon
        if isinstance(resolved_horizon, bool) or not isinstance(resolved_horizon, int):
            raise ValueError("horizon must be an integer")
        if resolved_horizon <= 0:
            raise ValueError("horizon must be positive")

        durations = {task.id: task.duration for task in self.tasks}
        earliest_finish: dict[str, int] = {}
        for node in nx.topological_sort(graph):
            start = max((earliest_finish[p] for p in graph.predecessors(node)), default=0)
            earliest_finish[node] = start + durations[node]
        critical_path = max(earliest_finish.values())
        if resolved_horizon < critical_path:
            raise ValueError(
                f"horizon {resolved_horizon} is below the precedence lower bound {critical_path}"
            )
        object.__setattr__(self, "horizon", resolved_horizon)

    @property
    def task_map(self) -> dict[str, Task]:
        return {task.id: task for task in self.tasks}

    @property
    def graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(task.id for task in self.tasks)
        graph.add_edges_from(self.precedence)
        return graph

    @property
    def resource_lower_bound(self) -> int:
        return ceil(sum(task.duration for task in self.tasks) / self.crew_limit)


@dataclass(frozen=True)
class Schedule:
    """Start times and actual completion time for every real task."""

    start_times: Mapping[str, int]
    makespan: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_times", dict(self.start_times))


@dataclass(frozen=True)
class SolverResult:
    """Common result envelope used by all solvers."""

    method: str
    schedule: Schedule | None
    feasible: bool
    energy: float | None
    runtime: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))


def validate_schedule(instance: ProjectInstance, schedule: Schedule) -> tuple[bool, list[str]]:
    """Independently check completeness, precedence, resources, and makespan."""

    errors: list[str] = []
    task_map = instance.task_map
    expected = set(task_map)
    supplied = set(schedule.start_times)
    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)
    if missing:
        errors.append(f"missing tasks: {missing}")
    if extra:
        errors.append(f"unknown tasks: {extra}")

    for task_id in expected & supplied:
        start = schedule.start_times[task_id]
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            errors.append(f"task {task_id} has invalid start {start!r}")

    if errors:
        return False, errors

    for predecessor, successor in instance.precedence:
        pred_finish = schedule.start_times[predecessor] + task_map[predecessor].duration
        if schedule.start_times[successor] < pred_finish:
            errors.append(f"precedence violated: {predecessor} -> {successor}")

    actual_makespan = max(
        schedule.start_times[task.id] + task.duration for task in instance.tasks
    )
    if schedule.makespan != actual_makespan:
        errors.append(
            f"makespan is {schedule.makespan}, but task completion implies {actual_makespan}"
        )
    if schedule.makespan > instance.horizon:
        errors.append(f"makespan {schedule.makespan} exceeds horizon {instance.horizon}")

    for time in range(actual_makespan):
        active = [
            task.id
            for task in instance.tasks
            if schedule.start_times[task.id] <= time
            < schedule.start_times[task.id] + task.duration
        ]
        if len(active) > instance.crew_limit:
            errors.append(
                f"crew capacity exceeded at time {time}: {len(active)} > {instance.crew_limit}"
            )

    return not errors, errors

