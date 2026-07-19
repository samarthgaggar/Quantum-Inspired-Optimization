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
