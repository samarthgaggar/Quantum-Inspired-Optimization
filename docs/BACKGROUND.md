# Background: RCPSP, CPM, QUBO, and Annealing

## Why scheduling becomes hard

If tasks only had precedence rules, earliest start times could be computed in a
single pass through a directed acyclic graph. Construction scheduling becomes
substantially harder when renewable resources are limited: two tasks might both
be precedence-ready while competing for the same crew. Every local choice can
delay a different downstream chain, so the number of possible orderings grows
combinatorially.

This simplified project is a single-mode **Resource-Constrained Project
Scheduling Problem (RCPSP)**:

- tasks are non-preemptive;
- durations are positive integers;
- each task requires one interchangeable crew;
- precedence relationships form a DAG; and
- at most `R` tasks may run at the same time.

General RCPSP is NP-hard. This does not mean that every small instance is hard;
it means there is no known algorithm guaranteed to solve all growing instances
in polynomial time.

## CPM is a lower bound, not the complete baseline

The Critical Path Method (CPM) calculates earliest and latest times while
ignoring crew contention. Its longest path is therefore a lower bound on the
makespan. A second lower bound is total work divided by crew capacity:

\[
L = \max\left(L_{CPM},\left\lceil \frac{\sum_i d_i}{R}\right\rceil\right).
\]

Raw CPM can schedule too many simultaneous tasks. The classical baseline in
this repository is a deterministic **parallel schedule-generation scheme**:
whenever a crew is free, it starts a precedence-ready task with the largest
remaining critical-path length. Task ID breaks ties. This returns a feasible
schedule while retaining CPM's urgency signal.

## From QUBO to an Ising energy landscape

A QUBO minimizes a quadratic polynomial over binary variables:

\[
E(x)=x^TQx,\qquad x_i\in\{0,1\}.
\]

Using the change of variables `x = (1 - s) / 2`, binary values map to Ising
spins `s` in `{-1,+1}`. The objective becomes a set of local fields and pairwise
spin couplings. Low-energy configurations represent good schedules; penalty
terms raise the energy of configurations that violate rules.

The time-indexed encoding is transparent but grows with both the task count and
time horizon. It is deliberately chosen for teaching, not claimed as the most
compact industrial RCPSP formulation.

## What simulated annealing does—and does not do

Simulated annealing is a classical Metropolis search. It begins at a high
temperature, where uphill moves are often accepted, and gradually cools so that
the search concentrates in lower-energy regions. Accepting temporary uphill
moves can escape local minima.

It is common in introductory QUBO projects to call this “quantum-inspired”
because it optimizes the same BQM/Ising representation used by quantum
annealers. Thermal simulated annealing does **not** model quantum tunnelling.
The maintained D-Wave Ocean interface is
`dwave.samplers.SimulatedAnnealingSampler`; the older `neal` namespace is
obsolete in current Ocean releases.

## QAOA comparison

The Quantum Approximate Optimization Algorithm (QAOA) alternates parameterized
cost and mixer unitaries, then uses a classical optimizer to tune circuit
angles. In this project it runs through Qiskit's state-vector simulator. Every
BQM variable becomes a qubit, and state-vector memory grows exponentially, so
QAOA is restricted to 2–4 task micro-instances with an explicit 18-qubit cap.
This demonstrates a gate-based workflow; it is not evidence of quantum speedup.

## Primary resources

- [D-Wave Ocean documentation](https://docs.dwavequantum.com/en/latest/ocean/)
- [dimod BQM reference](https://docs.dwavequantum.com/en/latest/ocean/api_ref_dimod/)
- [D-Wave simulated annealing API](https://docs.dwavequantum.com/en/latest/ocean/api_ref_samplers/generated/dwave.samplers.SimulatedAnnealingSampler.sample.html)
- [NetworkX DAG algorithms](https://networkx.org/documentation/stable/reference/algorithms/dag.html)
- [Qiskit Optimization QUBO/QAOA tutorial](https://qiskit-community.github.io/qiskit-optimization/tutorials/03_minimum_eigen_optimizer.html)
- [D-Wave job-shop scheduling example](https://github.com/dwave-examples/job-shop-scheduling)

