import pandas as pd
from matplotlib import pyplot as plt

from crew_schedule import build_qubo, construction_project, solve_cpm_list, toy_project
from crew_schedule.visualization import (
    plot_consistency,
    plot_dependency_graph,
    plot_gantt_comparison,
    plot_qaoa_scaling,
    plot_qubo_matrix,
    plot_scaling,
    plot_tuning,
    save_figure,
)


def test_problem_and_schedule_plots_render(tmp_path):
    project = construction_project()
    schedule = solve_cpm_list(project).schedule
    bqm, _ = build_qubo(toy_project())

    figures = [
        plot_dependency_graph(project)[0],
        plot_gantt_comparison(project, {"Classical": schedule})[0],
        plot_qubo_matrix(bqm)[0],
    ]
    for index, figure in enumerate(figures):
        output = tmp_path / f"figure_{index}.png"
        save_figure(figure, output)
        assert output.stat().st_size > 1_000
        plt.close(figure)


def test_experiment_plots_render_from_representative_tables():
    tuning = pd.DataFrame(
        [
            {"config": "quick", "feasible": True, "makespan": 5, "runtime": 0.1},
            {"config": "quality", "feasible": True, "makespan": 4, "runtime": 0.2},
        ]
    )
    benchmark = pd.DataFrame(
        [
            {
                "task_count": tasks,
                "crew_limit": crews,
                "annealing_feasible": True,
                "annealing_makespan": tasks + crews,
                "classical_makespan": tasks + crews - 1,
                "annealing_runtime": 0.01 * tasks,
                "classical_runtime": 0.0001 * tasks,
                "feasible_read_rate": 0.5,
            }
            for tasks in (4, 6)
            for crews in (2, 3)
        ]
    )
    qaoa = pd.DataFrame(
        [
            {"task_count": 2, "qubits": 8, "status": "completed", "reps": 1, "runtime": 0.2},
            {"task_count": 3, "qubits": 10, "status": "completed", "reps": 2, "runtime": 0.4},
            {
                "task_count": 10,
                "qubits": 93,
                "status": "not run—qubit limit",
                "reps": 1,
                "runtime": 0.0,
            },
        ]
    )

    figures = [
        plot_tuning(tuning)[0],
        plot_scaling(benchmark)[0],
        plot_consistency(benchmark)[0],
        plot_qaoa_scaling(qaoa)[0],
    ]
    for figure in figures:
        assert figure.axes
        plt.close(figure)
