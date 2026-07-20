# Mathematical Formulation and Four-Task Worked Example

## Time-indexed model

Let `i` index real tasks, `t` index integer time slots, `d_i` be a duration,
`R` be the number of crews, and `H` be a safe upper bound on makespan. The code
uses `H = sum(d_i)` unless the caller supplies a tighter valid horizon.

Binary variable `x[i,t]` equals one exactly when task `i` starts at `t`. The
builder first calculates precedence-only earliest and latest starts and creates
variables only inside those domains.

A zero-duration finish milestone `F` succeeds every terminal task. Its selected
start is the makespan objective:

\[
E_{objective}=\sum_{t\in T_F}t x_{F,t}.
\]

With penalty `A = H + 1`, the BQM is:

\[
E=E_{objective}+E_{once}+E_{precedence}+E_{crew}.
\]

### Start each task exactly once

\[
E_{once}=A\sum_i\left(\sum_{t\in T_i}x_{i,t}-1\right)^2.
\]

For two possible starts `a` and `b`, this expands using `x²=x` to
`A(-x_a-x_b+2x_ax_b+1)`. The interaction penalizes selecting both; the constant
and linear terms penalize selecting neither.

### Precedence

For every edge `i -> j`, any pair with `s < t + d_i` is forbidden:

\[
E_{precedence}=A\sum_{(i,j)}\sum_{t\in T_i}\sum_{s<t+d_i}x_{i,t}x_{j,s}.
\]

### Crew capacity

At time `tau`, the binary starts corresponding to running tasks form set
`C_tau`. Binary slack `z[tau,k]` converts the upper bound into an equality:

\[
E_{crew}=A\sum_\tau\left(
\sum_{(i,t)\in C_\tau}x_{i,t}
+\sum_k w_kz_{\tau,k}-R
\right)^2.
\]

`dimod.BinaryQuadraticModel.add_linear_inequality_constraint` generates the
minimum binary slack representation. The finish milestone consumes no crew.

Because the objective lies between zero and `H`, while every integer violation
contributes at least `A = H + 1`, no infeasible ground state can beat a feasible
schedule.

## Four-task construction example

The hand-verifiable example uses one-slot tasks:

| ID | Task | Predecessors |
|---|---|---|
| P | Site preparation | — |
| F | Foundation | P |
| U | Utility rough-in | P |
| I | Inspection | F, U |

There is one crew and `H=4`. The precedence-only critical path is three slots,
but total work divided by one crew is four, so the combined lower bound is four.
The pruned start domains are:

\[
T_P=\{0,1\},\quad T_F=T_U=\{1,2\},\quad
T_I=\{2,3\},\quad T_Finish=\{3,4\}.
\]

Here `A=5`. Selected examples of the explicit penalties are:

\[
5(x_{P,0}+x_{P,1}-1)^2
\]

for the one-hot site-preparation choice,

\[
5x_{P,1}x_{F,1}+5x_{P,1}x_{U,1}
\]

for two forbidden precedence pairs, and

\[
5(x_{P,1}+x_{F,1}+x_{U,1}+z_1-1)^2
\]

for crew capacity in slot one. Slot two has an analogous slack term. The finish
objective is `3*x[Finish,3] + 4*x[Finish,4]`.

After like terms are combined, the implemented BQM has 12 binary variables
(10 start variables and 2 slack variables), 18 quadratic interactions, and a
constant offset of 35. Exact enumeration finds energy four and either of the
equivalent crew-feasible schedules `P-F-U-I` or `P-U-F-I`. The notebook checks
this result before scaling to the ten-task project.

