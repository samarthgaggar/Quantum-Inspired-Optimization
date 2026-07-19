from pathlib import Path

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_delivered_notebook_is_executed_without_error_outputs():
    notebook = nbformat.read(ROOT / "notebooks" / "quantum_crew_scheduling.ipynb", as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells
    assert all(cell.execution_count is not None for cell in code_cells)
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert errors == []


def test_generated_results_cover_the_acceptance_matrix():
    benchmark = pd.read_csv(ROOT / "results" / "random_benchmarks.csv")
    qaoa = pd.read_csv(ROOT / "results" / "qaoa_micro_results.csv")
    summary = pd.read_csv(ROOT / "results" / "executive_summary.csv", index_col=0)["value"]
    assert len(benchmark) == 60
    assert benchmark["annealing_feasible"].all()
    assert set(benchmark["task_count"]) == {4, 6, 8, 10, 12}
    assert set(benchmark["crew_limit"]) == {2, 3}
    assert qaoa.loc[qaoa["task_count"] <= 4, "feasible"].all()
    assert "not run—qubit limit" in set(qaoa["status"])
    assert summary["classical_makespan"] == 20
    assert summary["annealing_makespan"] == 20


def test_expected_figures_exist_and_are_nonempty():
    expected = {
        "construction_dependencies.png",
        "toy_qubo_matrix.png",
        "annealing_tuning.png",
        "construction_gantt_comparison.png",
        "scaling_quality_runtime.png",
        "annealing_consistency.png",
        "qaoa_scaling.png",
    }
    for name in expected:
        path = ROOT / "figures" / name
        assert path.exists() and path.stat().st_size > 1_000

