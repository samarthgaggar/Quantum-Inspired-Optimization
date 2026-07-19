import itertools

import dimod
import numpy as np

from crew_schedule import build_qubo, decode_sample, encode_schedule, toy_project
from crew_schedule.classical import solve_cpm_list
from crew_schedule.examples import micro_projects
from crew_schedule.qaoa import bqm_to_quadratic_program


def test_domains_and_known_schedule_have_expected_energy():
    instance = toy_project()
    bqm, encoding = build_qubo(instance)
    assert encoding.domains["prep"] == (0, 1)
    assert bqm.num_variables <= 18
    schedule = solve_cpm_list(instance).schedule
    sample = encode_schedule(schedule, encoding)
    assert bqm.energy(sample) == schedule.makespan
    decoded, errors = decode_sample(sample, encoding)
    assert errors == []
    assert decoded == schedule


def test_exact_toy_ground_state_is_feasible_and_makespan_four():
    instance = toy_project()
    bqm, encoding = build_qubo(instance)
    result = dimod.ExactSolver().sample(bqm)
    schedule, errors = decode_sample(result.first.sample, encoding)
    assert errors == []
    assert schedule.makespan == 4
    assert result.first.energy == 4


def test_infeasible_micro_samples_cost_more_than_best_feasible():
    instance = micro_projects()[0]
    bqm, encoding = build_qubo(instance)
    best_feasible = float("inf")
    best_infeasible = float("inf")
    variables = list(bqm.variables)
    for bits in itertools.product((0, 1), repeat=len(variables)):
        sample = dict(zip(variables, bits, strict=True))
        energy = bqm.energy(sample)
        schedule, _ = decode_sample(sample, encoding)
        if schedule is None:
            best_infeasible = min(best_infeasible, energy)
        else:
            best_feasible = min(best_feasible, energy)
    assert best_feasible == 2
    assert best_infeasible > best_feasible


def test_qiskit_conversion_preserves_energy():
    bqm, _ = build_qubo(micro_projects()[0])
    program, variables = bqm_to_quadratic_program(bqm)
    rng = np.random.default_rng(7)
    for _ in range(10):
        bits = rng.integers(0, 2, len(variables))
        sample = dict(zip(variables, bits, strict=True))
        assert np.isclose(program.objective.evaluate(bits), bqm.energy(sample))
