import pandas as pd

from crew_schedule import solve_annealing
from crew_schedule.examples import micro_projects, toy_project
from crew_schedule.experiments import generate_random_project, run_benchmarks
from crew_schedule.models import validate_schedule
from crew_schedule.qaoa import solve_qaoa


def test_annealing_is_seeded_and_returns_valid_toy_schedule():
    first = solve_annealing(toy_project(), num_reads=40, num_sweeps=500, seed=91)
    second = solve_annealing(toy_project(), num_reads=40, num_sweeps=500, seed=91)
    assert first.feasible and second.feasible
    assert first.energy == second.energy
    assert first.schedule == second.schedule
    assert validate_schedule(toy_project(), first.schedule) == (True, [])


def test_random_generation_is_reproducible():
    first = generate_random_project(6, 2, seed=33)
    second = generate_random_project(6, 2, seed=33)
    assert first == second


def test_small_benchmark_schema_and_valid_results():
    frame = run_benchmarks(
        task_counts=(4,),
        crew_limits=(2,),
        instances_per_cell=1,
        repetitions=1,
        annealing_params={"num_reads": 20, "num_sweeps": 200},
    )
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 1
    assert {"gap_to_lower_bound", "bqm_variables", "feasible_read_rate"} <= set(frame)


def test_qaoa_cap_reports_skip_without_running_quantum_code():
    result = solve_qaoa(micro_projects()[0], qubit_cap=1)
    assert not result.feasible
    assert result.metadata["status"] == "not run—qubit limit"

