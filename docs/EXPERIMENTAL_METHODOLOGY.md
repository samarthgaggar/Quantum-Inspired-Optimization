# Experimental Methodology and Interpretation

## Fixed construction project

The main instance has ten construction activities, three interchangeable crews,
integer durations, and a fork/join precedence graph. Both solvers are checked by
an independent validator. Reported timing separates BQM construction from
sampling, while end-to-end time is used for method comparison.

## Annealing tuning

Four configurations span 50/200 reads, 500/2,000 sweeps, geometric/linear beta
schedules, and automatic/widened beta ranges. Each configuration uses seeds 101,
202, and 303. Selection is lexicographic:

1. highest fraction of runs returning at least one feasible schedule;
2. lowest median feasible makespan; then
3. lowest median runtime.

This avoids selecting a configuration from one lucky stochastic run.

## Random benchmark

The default matrix contains task counts 4, 6, 8, 10, and 12; crew limits 2 and
3; two generated DAGs per cell; and three annealing repetitions. That gives 60
annealing observations. Durations are uniform integers from 1 through 4. Every
task after the first receives one required earlier predecessor, and additional
forward edges appear with probability 0.15. All RNG seeds are recorded in CSV.

Metrics include makespan, combined lower-bound gap, difference from the
classical schedule, feasibility, feasible-read rate, build/sample/end-to-end
time, and BQM variables/interactions. Medians and interquartile ranges summarize
stochastic results.

## QAOA

The 2-, 3-, and 4-task micro-instances use depths `p=1,2`, seeds 11/22/33,
1,024 shots, and COBYLA capped at 100 evaluations. Qiskit's exact NumPy minimum
eigensolver verifies each BQM ground energy. Models above 18 variables are
marked `not run—qubit limit`; no cloud hardware or credentials are required.

## What the executed results show

- The fixed ten-task project has a combined lower bound of 20. Both the
  CPM-priority list scheduler and the best annealing run reach 20.
- All 60 randomized annealing runs decode to feasible schedules. Sixteen match
  the classical schedule and 44 are longer; none are shorter in this seeded
  sample. The result therefore supports feasibility, not quantum advantage.
- All 18 micro-QAOA runs contain a feasible decoded schedule. Some measured bit
  strings retain non-optimal slack bits and therefore have energy above the
  exact ground energy even when their decoded real-task schedule is optimal.
- The ten-task BQM needs 93 variables/qubits and is correctly skipped by the
  18-qubit local QAOA policy. Random 12-task BQMs reach up to 297 variables.

## Limitations

The benchmark is intentionally educational and too small for broad performance
claims. Crews are interchangeable; each task needs exactly one crew; durations
are deterministic; and there are no skills, calendars, costs, setup times,
preemption, uncertainty, or real quantum hardware. Runtime comparisons are
machine- and implementation-dependent. A production study should add a proven
optimal solver or stronger lower bounds, standard RCPSP benchmark sets, more
seeds, statistical tests, and richer resource models.

