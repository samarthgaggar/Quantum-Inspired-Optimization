"""Time-indexed QUBO construction and sample encoding/decoding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product

import dimod
import networkx as nx

from .models import ProjectInstance, Schedule, validate_schedule

FINISH_ID = "__finish__"


@dataclass(frozen=True)
class QuboEncoding:
    """Metadata required to interpret a scheduling BQM."""

    instance: ProjectInstance
    horizon: int
    penalty: float
    finish_id: str
    domains: Mapping[str, tuple[int, ...]]
    start_variables: Mapping[tuple[str, int], str]
    reverse_variables: Mapping[str, tuple[str, int]]
    resource_slacks: Mapping[int, tuple[tuple[str, int], ...]]

    @property
    def decision_variable_count(self) -> int:
        return len(self.start_variables)


def _variable_name(index: int, task_id: str, time: int) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in task_id)
    return f"x_{index:04d}_{safe}_{time}"


def _time_domains(
    instance: ProjectInstance,
) -> tuple[dict[str, tuple[int, ...]], list[tuple[str, str]]]:
    graph = instance.graph
    durations = {task.id: task.duration for task in instance.tasks}
    leaves = [node for node in graph if graph.out_degree(node) == 0]
    extended = graph.copy()
    extended.add_node(FINISH_ID)
    extended.add_edges_from((leaf, FINISH_ID) for leaf in leaves)
    durations[FINISH_ID] = 0

    order = list(nx.topological_sort(extended))
    earliest: dict[str, int] = {}
    for node in order:
        earliest[node] = max(
            (earliest[p] + durations[p] for p in extended.predecessors(node)), default=0
        )

    latest: dict[str, int] = {FINISH_ID: instance.horizon}
    for node in reversed(order[:-1]):
        latest[node] = min(
            latest[successor] - durations[node] for successor in extended.successors(node)
        )

    domains = {node: tuple(range(earliest[node], latest[node] + 1)) for node in order}
    if any(not domain for domain in domains.values()):
        raise ValueError("horizon leaves at least one task with no valid start-time domain")
    return domains, list(extended.edges())


def build_qubo(instance: ProjectInstance) -> tuple[dimod.BinaryQuadraticModel, QuboEncoding]:
    """Build the exact-makespan time-indexed scheduling BQM.

    Penalty ``H + 1`` dominates the non-negative finish-time objective, whose
    full range is 0 through H.
    """

    domains, extended_edges = _time_domains(instance)
    penalty = float(instance.horizon + 1)
    bqm = dimod.BinaryQuadraticModel("BINARY")
    start_variables: dict[tuple[str, int], str] = {}
    reverse_variables: dict[str, tuple[str, int]] = {}
    index = 0
    for task_id, domain in domains.items():
        for time in domain:
            name = _variable_name(index, task_id, time)
            index += 1
            start_variables[(task_id, time)] = name
            reverse_variables[name] = (task_id, time)

    # Every real task and the finish milestone starts exactly once.
    for task_id, domain in domains.items():
        bqm.add_linear_equality_constraint(
            [(start_variables[(task_id, time)], 1) for time in domain],
            lagrange_multiplier=penalty,
            constant=-1,
        )

    # A positive term penalizes every selected predecessor/successor pair that
    # violates successor_start >= predecessor_start + predecessor_duration.
    durations = {task.id: task.duration for task in instance.tasks}
    durations[FINISH_ID] = 0
    for predecessor, successor in extended_edges:
        for pred_time, succ_time in product(domains[predecessor], domains[successor]):
            if succ_time < pred_time + durations[predecessor]:
                bqm.add_quadratic(
                    start_variables[(predecessor, pred_time)],
                    start_variables[(successor, succ_time)],
                    penalty,
                )

    # At each slot, selected starts imply a set of running tasks. Slack turns
    # sum(active) <= crew_limit into a squared equality penalty.
    resource_slacks: dict[int, tuple[tuple[str, int], ...]] = {}
    for time in range(instance.horizon):
        active_terms: list[tuple[str, int]] = []
        for task in instance.tasks:
            for start in domains[task.id]:
                if start <= time < start + task.duration:
                    active_terms.append((start_variables[(task.id, start)], 1))
        if len(active_terms) <= instance.crew_limit:
            continue
        slack_terms = tuple(
            bqm.add_linear_inequality_constraint(
                active_terms,
                lagrange_multiplier=penalty,
                label=f"crew_{time}",
                lb=0,
                ub=instance.crew_limit,
            )
        )
        resource_slacks[time] = slack_terms

    # The selected finish-milestone time is the makespan objective.
    for time in domains[FINISH_ID]:
        bqm.add_linear(start_variables[(FINISH_ID, time)], float(time))

    encoding = QuboEncoding(
        instance=instance,
        horizon=instance.horizon,
        penalty=penalty,
        finish_id=FINISH_ID,
        domains={task_id: tuple(domain) for task_id, domain in domains.items()},
        start_variables=start_variables,
        reverse_variables=reverse_variables,
        resource_slacks=resource_slacks,
    )
    return bqm, encoding


def decode_sample(
    sample: Mapping[str, int | float], encoding: QuboEncoding
) -> tuple[Schedule | None, list[str]]:
    """Decode a bit sample, rejecting ambiguous or invalid schedules."""

    errors: list[str] = []
    selected: dict[str, int] = {}
    for task_id, domain in encoding.domains.items():
        starts = [
            time
            for time in domain
            if float(sample.get(encoding.start_variables[(task_id, time)], 0)) > 0.5
        ]
        if len(starts) != 1:
            errors.append(f"{task_id} selected {len(starts)} start times")
        else:
            selected[task_id] = starts[0]
    if errors:
        return None, errors

    finish_time = selected.pop(encoding.finish_id)
    actual_makespan = max(selected[task.id] + task.duration for task in encoding.instance.tasks)
    if finish_time < actual_makespan:
        errors.append(
            f"finish milestone {finish_time} precedes actual completion {actual_makespan}"
        )
        return None, errors
    schedule = Schedule(selected, actual_makespan)
    feasible, validation_errors = validate_schedule(encoding.instance, schedule)
    if not feasible:
        return None, validation_errors
    return schedule, []


def analyze_sample(sample: Mapping[str, int | float], encoding: QuboEncoding) -> dict[str, object]:
    """Explain decision-level violations in a BQM sample.

    Slack-bit consistency remains represented by the BQM energy itself. This
    report focuses on the scheduling decisions a reader can interpret directly.
    """

    selected = {
        task_id: [
            time
            for time in domain
            if float(sample.get(encoding.start_variables[(task_id, time)], 0)) > 0.5
        ]
        for task_id, domain in encoding.domains.items()
    }
    one_hot_deviation = {
        task_id: abs(len(times) - 1) for task_id, times in selected.items() if len(times) != 1
    }

    durations = {task.id: task.duration for task in encoding.instance.tasks}
    precedence_violations: list[tuple[str, str, int, int]] = []
    for predecessor, successor in encoding.instance.precedence:
        for pred_start, succ_start in product(selected[predecessor], selected[successor]):
            if succ_start < pred_start + durations[predecessor]:
                precedence_violations.append((predecessor, successor, pred_start, succ_start))

    crew_usage: dict[int, int] = {}
    crew_overload: dict[int, int] = {}
    for time in range(encoding.horizon):
        usage = sum(
            start <= time < start + durations[task.id]
            for task in encoding.instance.tasks
            for start in selected[task.id]
        )
        crew_usage[time] = usage
        if usage > encoding.instance.crew_limit:
            crew_overload[time] = usage - encoding.instance.crew_limit

    finish_times = selected[encoding.finish_id]
    terminal_violations: list[tuple[str, int, int]] = []
    if len(finish_times) == 1:
        finish_time = finish_times[0]
        graph = encoding.instance.graph
        for terminal in (node for node in graph if graph.out_degree(node) == 0):
            for start in selected[terminal]:
                if finish_time < start + durations[terminal]:
                    terminal_violations.append((terminal, start, finish_time))

    return {
        "selected_starts": selected,
        "one_hot_deviation": one_hot_deviation,
        "precedence_violations": precedence_violations,
        "terminal_violations": terminal_violations,
        "crew_usage": crew_usage,
        "crew_overload": crew_overload,
        "finish_objective": sum(finish_times),
        "decision_feasible": not (
            one_hot_deviation or precedence_violations or terminal_violations or crew_overload
        ),
    }


def _subset_bits(weighted_variables: tuple[tuple[str, int], ...], target: int) -> dict[str, int]:
    variables = list(weighted_variables)
    for bits in product((0, 1), repeat=len(variables)):
        if sum(bit * weight for bit, (_, weight) in zip(bits, variables, strict=True)) == target:
            return {name: bit for bit, (name, _) in zip(bits, variables, strict=True)}
    raise ValueError(f"slack value {target} cannot be represented")


def encode_schedule(schedule: Schedule, encoding: QuboEncoding) -> dict[str, int]:
    """Encode a known feasible schedule, including capacity slack variables."""

    feasible, errors = validate_schedule(encoding.instance, schedule)
    if not feasible:
        raise ValueError(f"cannot encode invalid schedule: {errors}")
    sample = {variable: 0 for variable in encoding.reverse_variables}
    for task_id, start in schedule.start_times.items():
        variable = encoding.start_variables.get((task_id, start))
        if variable is None:
            raise ValueError(f"start {start} for {task_id} is outside the QUBO domain")
        sample[variable] = 1
    finish_variable = encoding.start_variables.get((encoding.finish_id, schedule.makespan))
    if finish_variable is None:
        raise ValueError("schedule makespan is outside the finish domain")
    sample[finish_variable] = 1

    task_map = encoding.instance.task_map
    for time, slack_terms in encoding.resource_slacks.items():
        active = sum(
            schedule.start_times[task_id]
            <= time
            < schedule.start_times[task_id] + task_map[task_id].duration
            for task_id in schedule.start_times
        )
        sample.update(_subset_bits(slack_terms, encoding.instance.crew_limit - active))
    return sample
