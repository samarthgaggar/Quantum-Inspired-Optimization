# Quantum-Inspired Optimization for Construction Crew Scheduling

This project turns a small resource-constrained construction schedule into a
time-indexed Quadratic Unconstrained Binary Optimization (QUBO) model. It
compares three approaches:

1. a deterministic CPM-priority list scheduler;
2. D-Wave's local, classical simulated annealer; and
3. QAOA on a local state-vector simulator for micro-instances that fit within
   an 18-qubit limit.

The wording matters: CPM by itself ignores crew capacity, so it is reported as
a lower bound rather than as a feasible resource-constrained schedule.
Simulated annealing is a thermal classical heuristic. It is often described as
"quantum-inspired" in introductory projects because it operates on the same
QUBO/Ising energy landscape used by quantum annealers, but it does not simulate
quantum tunnelling. QAOA is a quantum algorithm, although this project executes
it on a classical simulator and makes no claim of quantum advantage.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
pytest
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/quantum_crew_scheduling.ipynb \
  --ExecutePreprocessor.timeout=900
```

The executed notebook writes tables to `results/` and figures to `figures/`.
The default experiment profile is intentionally compact. The reusable
`run_benchmarks` function can be given more instances, repetitions, reads, or
sweeps for an extended run.

## Model

Each real task has an integer duration, uses one interchangeable crew, cannot
be preempted, and must respect a directed acyclic precedence graph. Binary
variable `x[i,t]` is one when task `i` starts at time `t`. A zero-duration finish
milestone makes the objective linear. One-hot, precedence, and per-time-slot
crew-capacity rules are converted into quadratic penalties.

The package lives in `src/crew_schedule`; the notebook is the guided report,
while the modules and tests keep its computations reusable and verifiable.

## What is included

- A hand-verifiable four-task QUBO whose exact ground state has makespan four.
- A ten-task construction project with site preparation, foundation, framing,
  roofing, plumbing, electrical, insulation, drywall, interior finishing, and
  inspection.
- Earliest/latest CPM calculations, a combined precedence/resource lower bound,
  and a resource-feasible CPM-priority list scheduler.
- A reusable `dimod.BinaryQuadraticModel` builder with one-hot, precedence, and
  crew-capacity penalties plus independent schedule decoding and validation.
- Seeded simulated-annealing tuning over reads, sweeps, beta ranges, and schedule
  types.
- A 60-run random-instance benchmark with quality, runtime, feasibility,
  consistency, and model-size metrics.
- Local QAOA at depths 1 and 2 for 2–4 task models, exact eigensolver checks,
  and an explicit 18-qubit cap.
- An executed notebook, CSV result tables, dependency/QUBO/Gantt/scaling plots,
  automated tests, coverage configuration, linting, and GitHub Actions CI.

The executed study finds a 20-slot schedule from both the classical and
annealing methods on the fixed project. Across the 60 seeded randomized runs,
annealing returns a feasible schedule every time, matches the classical
schedule 16 times, and is longer 44 times. These results are intentionally
reported without a quantum-advantage claim.

## Project guide

- [`notebooks/quantum_crew_scheduling.ipynb`](notebooks/quantum_crew_scheduling.ipynb)
  is the complete, executable report.
- [`docs/BACKGROUND.md`](docs/BACKGROUND.md) explains RCPSP, CPM, Ising/QUBO,
  simulated annealing, and QAOA.
- [`docs/MATHEMATICAL_FORMULATION.md`](docs/MATHEMATICAL_FORMULATION.md) derives
  the penalties and works through the four-task example.
- [`docs/EXPERIMENTAL_METHODOLOGY.md`](docs/EXPERIMENTAL_METHODOLOGY.md) records
  seeds, tuning policy, metrics, findings, and limitations.
- `src/crew_schedule/` contains the reusable implementation; `tests/` contains
  unit, integration, energy-equivalence, and delivered-artifact checks.

## Quality checks

```bash
ruff check .
ruff format --check .
pytest --cov=crew_schedule --cov-report=term-missing
```

The GitHub Actions workflow runs these checks on Python 3.10 and 3.13. Notebook
execution is kept as an explicit reproducibility step because the full local
QAOA matrix is much slower than the unit-test suite.

## Primary references

- [D-Wave Ocean documentation](https://docs.dwavequantum.com/en/latest/ocean/)
- [dimod BQM documentation](https://docs.dwavequantum.com/en/latest/ocean/api_ref_dimod/)
- [D-Wave simulated annealing API](https://docs.dwavequantum.com/en/latest/ocean/api_ref_samplers/generated/dwave.samplers.SimulatedAnnealingSampler.sample.html)
- [NetworkX DAG algorithms](https://networkx.org/documentation/stable/reference/algorithms/dag.html)
- [Qiskit Optimization QAOA tutorial](https://qiskit-community.github.io/qiskit-optimization/tutorials/03_minimum_eigen_optimizer.html)
