"""Qiskit conversion, exact micro-instance verification, and local QAOA."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

import numpy as np

from .models import ProjectInstance, SolverResult
from .qubo import build_qubo, decode_sample

_EXACT_ENERGY_CACHE: dict[tuple, float] = {}


def _bqm_fingerprint(bqm) -> tuple:
    linear = tuple(sorted((str(variable), float(bias)) for variable, bias in bqm.linear.items()))
    quadratic = tuple(
        sorted(
            (str(left), str(right), float(bias)) for (left, right), bias in bqm.quadratic.items()
        )
    )
    return float(bqm.offset), linear, quadratic


def bqm_to_quadratic_program(bqm):
    """Convert a dimod BQM into an equivalent unconstrained QuadraticProgram."""

    from qiskit_optimization import QuadraticProgram

    variables = list(bqm.variables)
    safe_names = {variable: f"v{index:04d}" for index, variable in enumerate(variables)}
    program = QuadraticProgram("scheduling_qubo")
    for name in safe_names.values():
        program.binary_var(name=name)
    linear = {safe_names[variable]: float(bias) for variable, bias in bqm.linear.items()}
    quadratic = {
        (safe_names[left], safe_names[right]): float(bias)
        for (left, right), bias in bqm.quadratic.items()
    }
    program.minimize(constant=float(bqm.offset), linear=linear, quadratic=quadratic)
    return program, variables


def solve_qaoa(
    instance: ProjectInstance,
    *,
    reps: int = 1,
    seed: int = 123,
    shots: int = 1_024,
    maxiter: int = 100,
    qubit_cap: int = 18,
) -> SolverResult:
    """Solve a qubit-feasible BQM using local state-vector QAOA."""

    started = perf_counter()
    bqm, encoding = build_qubo(instance)
    if bqm.num_variables > qubit_cap:
        return SolverResult(
            method=f"QAOA p={reps}",
            schedule=None,
            feasible=False,
            energy=None,
            runtime=perf_counter() - started,
            metadata={
                "status": "not run—qubit limit",
                "qubits": bqm.num_variables,
                "qubit_cap": qubit_cap,
                "reps": reps,
                "seed": seed,
            },
        )

    from qiskit.primitives import StatevectorSampler
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_optimization.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
    from qiskit_optimization.optimizers import COBYLA
    from qiskit_optimization.utils import algorithm_globals

    program, variables = bqm_to_quadratic_program(bqm)
    algorithm_globals.random_seed = seed
    fingerprint = _bqm_fingerprint(bqm)
    if fingerprint not in _EXACT_ENERGY_CACHE:
        exact_result = MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(program)
        _EXACT_ENERGY_CACHE[fingerprint] = float(exact_result.fval)
    exact_ground_energy = _EXACT_ENERGY_CACHE[fingerprint]

    rng = np.random.default_rng(seed)
    initial_point = rng.uniform(-np.pi, np.pi, size=2 * reps)
    sampler = StatevectorSampler(default_shots=shots, seed=seed)
    qaoa = QAOA(
        sampler=sampler,
        optimizer=COBYLA(maxiter=maxiter),
        reps=reps,
        initial_point=initial_point,
    )
    result = MinimumEigenOptimizer(qaoa).solve(program)

    best_schedule = None
    best_energy = None
    best_probability = None
    feasible_probability = 0.0
    result_samples: Sequence = result.samples or ()
    for candidate in sorted(result_samples, key=lambda sample: sample.fval):
        sample = {
            variable: int(round(float(bit)))
            for variable, bit in zip(variables, candidate.x, strict=True)
        }
        schedule, _ = decode_sample(sample, encoding)
        if schedule is None:
            continue
        feasible_probability += float(candidate.probability)
        if best_schedule is None:
            best_schedule = schedule
            best_energy = float(candidate.fval)
            best_probability = float(candidate.probability)

    return SolverResult(
        method=f"QAOA p={reps}",
        schedule=best_schedule,
        feasible=best_schedule is not None,
        energy=best_energy,
        runtime=perf_counter() - started,
        metadata={
            "status": "completed" if best_schedule else "no feasible measured sample",
            "qubits": bqm.num_variables,
            "qubit_cap": qubit_cap,
            "reps": reps,
            "seed": seed,
            "shots": shots,
            "maxiter": maxiter,
            "exact_ground_energy": exact_ground_energy,
            "best_probability": best_probability,
            "feasible_probability": feasible_probability,
        },
    )
