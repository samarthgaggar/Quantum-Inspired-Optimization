import pytest

from crew_schedule import ProjectInstance, Schedule, Task, construction_project, toy_project
from crew_schedule.classical import cpm_analysis, solve_cpm_list
from crew_schedule.models import validate_schedule


def test_project_validation_rejects_bad_inputs():
    with pytest.raises(ValueError, match="positive integer"):
        Task("a", "A", 0)
    with pytest.raises(ValueError, match="acyclic"):
        ProjectInstance(
            (Task("a", "A", 1), Task("b", "B", 1)),
            (("a", "b"), ("b", "a")),
            1,
        )
    with pytest.raises(ValueError, match="unknown task"):
        ProjectInstance((Task("a", "A", 1),), (("a", "b"),), 1)


def test_toy_classical_schedule_is_feasible_and_optimal():
    instance = toy_project()
    result = solve_cpm_list(instance)
    assert result.feasible
    assert result.schedule.makespan == 4
    analysis = cpm_analysis(instance)
    assert analysis["critical_path_length"] == 3
    assert analysis["combined_lower_bound"] == 4
    assert validate_schedule(instance, result.schedule) == (True, [])


def test_schedule_validator_catches_capacity_and_makespan():
    instance = toy_project()
    schedule = Schedule(
        {"prep": 0, "foundation": 1, "utilities": 1, "inspection": 2},
        3,
    )
    feasible, errors = validate_schedule(instance, schedule)
    assert not feasible
    assert any("capacity" in error for error in errors)


def test_main_classical_schedule_is_feasible():
    result = solve_cpm_list(construction_project())
    assert result.feasible
    assert result.schedule is not None
