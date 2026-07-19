"""Plotting helpers for the notebook report."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .models import ProjectInstance, Schedule


def plot_dependency_graph(instance: ProjectInstance):
    graph = instance.graph
    generations = list(nx.topological_generations(graph))
    positions: dict[str, tuple[float, float]] = {}
    for x, generation in enumerate(generations):
        for y, node in enumerate(generation):
            positions[node] = (x, -y)
    labels = {task.id: task.name for task in instance.tasks}
    fig, ax = plt.subplots(figsize=(11, 5))
    nx.draw_networkx(
        graph,
        pos=positions,
        labels=labels,
        node_color="#dbeafe",
        edge_color="#64748b",
        node_size=2_200,
        font_size=8,
        arrowsize=18,
        ax=ax,
    )
    ax.set_title(f"Dependency graph: {instance.name}")
    ax.axis("off")
    fig.tight_layout()
    return fig, ax


def _draw_gantt(ax, instance: ProjectInstance, schedule: Schedule, title: str) -> None:
    task_map = instance.task_map
    ordered = sorted(
        instance.tasks,
        key=lambda task: (schedule.start_times[task.id], task.id),
        reverse=True,
    )
    colors = plt.cm.Set3(np.linspace(0, 1, len(ordered)))
    for row, (task, color) in enumerate(zip(ordered, colors, strict=True)):
        start = schedule.start_times[task.id]
        ax.barh(row, task.duration, left=start, color=color, edgecolor="#334155")
        ax.text(start + task.duration / 2, row, task.id, ha="center", va="center", fontsize=8)
    ax.set_yticks(range(len(ordered)), [task_map[task.id].name for task in ordered])
    ax.set_xlabel("Time slot")
    ax.set_xlim(0, max(schedule.makespan + 1, 2))
    ax.grid(axis="x", alpha=0.25)
    ax.set_title(f"{title} — makespan {schedule.makespan}")


def plot_gantt_comparison(
    instance: ProjectInstance,
    schedules: Mapping[str, Schedule],
):
    fig, axes = plt.subplots(
        1,
        len(schedules),
        figsize=(7 * len(schedules), 5),
        squeeze=False,
        sharex=True,
    )
    for ax, (label, schedule) in zip(axes[0], schedules.items(), strict=True):
        _draw_gantt(ax, instance, schedule, label)
    fig.tight_layout()
    return fig, axes


def plot_qubo_matrix(bqm):
    variables = list(bqm.variables)
    index = {variable: position for position, variable in enumerate(variables)}
    matrix = np.zeros((len(variables), len(variables)))
    for variable, bias in bqm.linear.items():
        matrix[index[variable], index[variable]] = bias
    for (left, right), bias in bqm.quadratic.items():
        matrix[index[left], index[right]] = bias
        matrix[index[right], index[left]] = bias
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
    ax.set_title(f"QUBO coefficient matrix ({len(variables)} variables)")
    ax.set_xlabel("Binary variable")
    ax.set_ylabel("Binary variable")
    fig.colorbar(image, ax=ax, shrink=0.8, label="Bias")
    fig.tight_layout()
    return fig, ax


def plot_tuning(tuning: pd.DataFrame):
    summary = tuning.groupby("config", sort=False).agg(
        success=("feasible", "mean"),
        makespan=("makespan", "median"),
        runtime=("runtime", "median"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    summary["success"].plot.bar(ax=axes[0], color="#2563eb", title="Feasible run rate")
    summary["makespan"].plot.bar(ax=axes[1], color="#16a34a", title="Median makespan")
    summary["runtime"].plot.bar(ax=axes[2], color="#f97316", title="Median runtime")
    axes[0].set_ylim(0, 1.05)
    axes[2].set_ylabel("Seconds")
    for ax in axes:
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Simulated-annealing parameter sensitivity")
    fig.tight_layout()
    return fig, axes


def plot_scaling(benchmark: pd.DataFrame):
    feasible = benchmark[benchmark["annealing_feasible"]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for crew_limit, subset in feasible.groupby("crew_limit"):
        grouped = subset.groupby("task_count")["annealing_makespan"]
        medians = grouped.median()
        lower = grouped.quantile(0.25)
        upper = grouped.quantile(0.75)
        axes[0].plot(medians.index, medians.values, marker="o", label=f"SA, {crew_limit} crews")
        axes[0].fill_between(medians.index, lower.values, upper.values, alpha=0.15)
    classical = benchmark.groupby(["task_count", "crew_limit"])["classical_makespan"].median()
    for crew_limit in sorted(benchmark["crew_limit"].unique()):
        values = classical.xs(crew_limit, level="crew_limit")
        axes[0].plot(values.index, values.values, linestyle="--", label=f"Classical, {crew_limit} crews")

    runtime = benchmark.groupby("task_count").agg(
        annealing=("annealing_runtime", "median"),
        classical=("classical_runtime", "median"),
    )
    axes[1].plot(runtime.index, runtime["annealing"], marker="o", label="Simulated annealing")
    axes[1].plot(runtime.index, runtime["classical"], marker="o", label="Classical")
    axes[1].set_yscale("log")
    axes[0].set_ylabel("Makespan")
    axes[1].set_ylabel("Median runtime (seconds, log scale)")
    for ax in axes:
        ax.set_xlabel("Number of tasks")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Solution quality and runtime scaling")
    fig.tight_layout()
    return fig, axes


def plot_consistency(benchmark: pd.DataFrame):
    grouped = benchmark.groupby("task_count").agg(
        feasible_run_rate=("annealing_feasible", "mean"),
        median_feasible_read_rate=("feasible_read_rate", "median"),
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    grouped.plot.bar(ax=ax, color=["#7c3aed", "#06b6d4"])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Number of tasks")
    ax.set_ylabel("Rate")
    ax.set_title("Annealing feasibility and consistency")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_qaoa_scaling(qaoa_results: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    qubits = qaoa_results.groupby("task_count")["qubits"].max()
    axes[0].plot(qubits.index, qubits.values, marker="o", color="#9333ea")
    axes[0].axhline(18, color="#dc2626", linestyle="--", label="Local cap")
    axes[0].set_ylabel("Qubits / BQM variables")
    axes[0].legend()
    completed = qaoa_results[qaoa_results["status"] == "completed"]
    for reps, subset in completed.groupby("reps"):
        medians = subset.groupby("task_count")["runtime"].median()
        axes[1].plot(medians.index, medians.values, marker="o", label=f"p={reps}")
    axes[1].set_ylabel("Median QAOA runtime (seconds)")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("Real tasks")
        ax.grid(alpha=0.25)
    fig.suptitle("Why local QAOA is limited to micro-instances")
    fig.tight_layout()
    return fig, axes


def save_figure(fig, path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight")

