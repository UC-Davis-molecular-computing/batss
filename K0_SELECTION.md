# Choosing the filler count $k_0$

How `batss` sizes the filler/catalyst species $\mathbf{K}$ that pads every reaction up to the CRN's
order. This supersedes the old "reset $\mathbf{K}$ to $n$" heuristic. The claims here are all
confirmed empirically (see [Empirical validation](#empirical-validation)).

## Notation

| symbol | meaning |
|---|---|
| $n$ | number of *real* molecules (sum of the original-species counts) |
| $k_0$ | number of filler molecules $\mathbf{K}$ (the knob we choose) |
| $N = n + k_0$ | total population the sampler draws from |
| $o$ | order of the CRN, i.e. the largest reactant count of any reaction |
| $a$ | original order of one reaction (before padding); $\delta_0 = o - a$ fillers are added to it |
| $v$ | volume; rate constants are given in the deterministic/macroscopic convention, so an $a$-reactant reaction's rate is divided by $v^{a-1}$ |
| $p$ | probability a sampled $o$-molecule set fires a *real* (non-passive) reaction |
| $\mathbb{E}[\ell]$ | expected collision-free run length: reactions per batch before a collision |
| $k_{\max}$ | largest adjusted rate constant over all reactions (`continuous_time_correction_factor`) |

## The objective: maximize $\mathbb{E}[\ell] \cdot p$

Each batch costs one expensive collision-length sample and accomplishes $\mathbb{E}[\ell] \cdot p$ real
reactions, so to simulate a fixed number $R$ of real reactions,

$$
\#\mathrm{batches} = \frac{R}{\mathbb{E}[\ell] \cdot p},
\qquad
R \text{ independent of } k_0.
$$

Minimizing the batch count is exactly **maximizing $\mathbb{E}[\ell] \cdot p$**. Two facts make this
tractable:

- $\mathbb{E}[\ell] = c\sqrt{N} + o(\sqrt{n}) \approx c\sqrt{N}$, with
  $c = \sqrt{\pi/(2o(o+g))}$ a constant fixed by the CRN. Here $g$ is the CRN's "generativity";
  $g = 1$ for LV/Rössler and $g = 2$ for Oregonator. The constant $c$ does not depend on $k_0$.
- $p = \dfrac{P_{\mathrm{real}}}{k_{\max}\binom{N}{o}} \approx \dfrac{P_{\mathrm{real}}}{k_{\max} N^o}$, and $P_{\mathrm{real}}$ is independent of $k_0$:
  Definition 2.7's rate correction divides a padded reaction's constant by the $k_0$ falling
  factorial of its $\mathbf{K}$-multiplicity, exactly canceling the extra ways a padded reaction can
  draw its $\mathbf{K}$ molecules from the $k_0$ available. The real dynamics are therefore
  filler-invariant.

Dropping the factors that are independent of $k_0$,

$$
\argmax_{k_0}\ \mathbb{E}[\ell] \cdot p
\quad=\quad
\argmin_{k_0}\ k_{\max}(k_0) \cdot (n+k_0)^{\,o - 1/2}.
$$

Contrast the paper's "simulation slowdown factor" $S = 1/p$ (Def. 2.8): minimizing $S$ maximizes $p$
alone, i.e. it minimizes $k_{\max}N^o$, dropping the $\sqrt{N} = \mathbb{E}[\ell]$ factor. That targets
$\min(n,\mathrm{crossover})$ instead of $\min(2n,\mathrm{crossover})$, which is the same answer only
when $n \ge \mathrm{crossover}$.

## $k_{\max}$ and the crossover

$k_{\max}$ is the largest adjusted rate constant. In general there is **one branch per reaction
order**, not two. Order-$a$ reactions all carry

$$
\delta = o - a
$$

fillers and so share the factor $1/k_0^{\,o-a}$. That means the largest adjusted rate *within* order
$a$ is just its largest

$$
A_a = \frac{\mathrm{rate}}{v^{a-1}} \cdot \mathrm{symmetry},
$$

which is independent of $k_0$, and order $a$ contributes the single branch
$A_a/k_0^{\,o-a}$. Hence

$$
k_{\max}(k_0) = \max_a \frac{A_a}{k_0^{\,o-a}}
\qquad
\text{(one term per order } a \text{ present).}
$$

This is a max whose winning order shifts as $k_0$ grows: each branch has a different slope $o-a$, and
the order-$o$ branch is the only branch independent of $k_0$. The general treatment (several branches,
several crossovers) is in **General CRNs** below.

**For $o = 2$**, which covers every benchmark CRN, only orders $1$ and $2$ occur, so there are exactly
two branches:

- **order 1** (padded, $\delta = 1$): $C_{\mathrm{padded}}/k_0$, with
  $C_{\mathrm{padded}} = \max_{\mathrm{order}\ 1}(\mathrm{rate}\cdot\mathrm{symmetry})$.
  Order-1 reactions get no volume division, and this branch falls with $k_0$.
- **order 2** ($a = o$, $\delta = 0$, unpadded): the constant
  $C_{\mathrm{flat}} = \max_{\mathrm{order}\ 2}(\mathrm{rate}/v\cdot\mathrm{symmetry})$.

So

$$
k_{\max}(k_0) = \max\left(\frac{C_{\mathrm{padded}}}{k_0}, C_{\mathrm{flat}}\right)
$$

rides the falling order-1 branch, then flattens onto $C_{\mathrm{flat}}$. The **crossover** where the
branches meet is

$$
\mathrm{crossover} = \frac{C_{\mathrm{padded}}}{C_{\mathrm{flat}}}.
$$

Because order-1 reactions get no volume division while order-2 reactions are divided by $v$,

$$
\mathrm{crossover}
= v \cdot
\frac{\max \mathrm{\ unimolecular\ rate}}
     {\max \mathrm{\ bimolecular\ rate}\cdot\mathrm{symmetry}},
$$

a **config-independent** quantity: it depends on rate constants and volume, not on $n$. The
rate-constant ratio is the CRN-specific multiple of $v$:

| CRN | max unimol. | max bimol. $\cdot$ sym | crossover |
|---|---:|---:|---:|
| Lotka-Volterra | $1$ | $1$ | $v$ |
| Oregonator | $520$ | $1000$ | $0.52v$ |
| Rössler-Willamowski | $30$ | $1$ | $30v$ |

Computed in `UniformCRN::crossover_k_count`.

## The optimum: $k_0^\star = \min(2n,\mathrm{crossover})$

Feed each branch of $k_{\max}$ into the objective
$k_{\max} \cdot (n+k_0)^{\,o - 1/2}$:

- **below the crossover**,
  $k_{\max}=C_{\mathrm{padded}}/k_0$, so the objective is proportional to
  $(n+k_0)^{\,o - 1/2}/k_0$. This decreases with $k_0$ up to the interior optimum
  $k_0 = n/(o - 3/2)$, which is $2n$ for $o = 2$.
- **above the crossover**, $k_{\max}=C_{\mathrm{flat}}$, so the objective is proportional to
  $(n+k_0)^{\,o - 1/2}$, which only increases.

So the batch count is minimized at

$$
k_0^\star
= \min\left(\frac{n}{o - 3/2},\ \mathrm{crossover}\right)
= \min(2n,\ \mathrm{crossover})
\qquad
\text{for } o = 2.
$$

Two regimes:

- **$\mathrm{crossover} \le 2n$** (LV, Oregonator): the crossover binds.
  $k_0^\star$ is **config-independent** and proportional to $v$, so it can be computed once from the
  reaction table and never revisited.
- **$\mathrm{crossover} > 2n$** (Rössler, whose fast autocatalysis makes
  $\mathrm{crossover}=30v$): the $2n$ term binds, so $k_0^\star$ tracks the population. As Rössler's
  $n$ explodes past $\mathrm{crossover}/2$, $k_0^\star$ rises to the crossover and the cap takes over.

## General CRNs (any order $o$)

The order-2 result is a special case. In general each reaction $j$ has adjusted rate

$$
\frac{A_j}{\operatorname{ff}(k_0,\delta_j)},
$$

where

$$
A_j =
(\text{volume-adjusted rate constant})\cdot(\text{symmetry degree}),
\qquad
\delta_j = o - \operatorname{order}(j),
$$

and $\operatorname{ff}(k_0,\delta)$ is the $\mathbf{K}$ falling factorial

$$
\operatorname{ff}(k_0,\delta)
= k_0(k_0-1)\cdots(k_0-\delta+1),
\qquad
\operatorname{ff}(k_0,0)=1,
\qquad
\operatorname{ff}(k_0,\delta)\approx k_0^\delta
\text{ when } k_0 \gg \delta.
$$

Then

$$
k_{\max}(k_0)
= \max_j \frac{A_j}{\operatorname{ff}(k_0,\delta_j)},
\qquad
k_0^\star
= \arg\min_{k_0 \ge 1}
k_{\max}(k_0)(n+k_0)^{\,o - 1/2}.
$$

Because $k_{\max}$ is a max of terms each proportional to $k_0^{-\delta_j}$, the objective is
piecewise smooth and its minimizer is the smallest-objective candidate among the points below.

**Structure of $k_{\max}$.** In log-log space each reaction is a straight line

$$
\log A_j - \delta_j \log k_0
$$

with slope $-\delta_j$, and $k_{\max}(k_0)$ is their upper envelope. As $k_0$ grows, the active
(topmost) line has **strictly decreasing** $\delta$: the steepest padded reaction (largest $\delta$)
wins at small $k_0$, successively flatter ones take over, and the flat floor

$$
C_{\mathrm{flat}} = \max_{\delta_j = 0} A_j
$$

wins at large $k_0$. Each adjacent pair of active lines meets at **one crossover**, so in general there
are *several* crossovers, not one.

**Candidates for the minimizer** of

$$
f(k_0) = k_{\max}(k_0) \cdot (n+k_0)^{o - 1/2}
$$

are:

- **the interior optimum of each active branch $j$**,

  $$
  k_0 = \frac{\delta_j n}{o - 1/2 - \delta_j},
  $$

  but only when it lands inside that branch's own $k_0$ interval (between its two crossovers) and
  $o - 1/2 - \delta_j > 0$. Otherwise $f$ is monotonic across that branch and it contributes no
  interior point.
- **each crossover kink**, since $f$ is continuous but bends there and its minimum can sit exactly on
  a kink: the left branch is still falling, while the right branch is already rising.

Evaluate $f$ at every valid candidate and take the smallest.

**So it is *not*
$\min(n/(o-3/2),\mathrm{crossover}_1,\mathrm{crossover}_2,\ldots)$.** The interior optimum is not a
fixed $n/(o-3/2)$ shared by all branches. It is

$$
\frac{\delta_j n}{o - 1/2 - \delta_j},
$$

which changes with the branch. For example, at $o=3$ it is $2n/3$ on the $\delta=1$ branch but $4n$
on the $\delta=2$ branch. And only the interiors that fall in range enter the minimum, next to the
several crossovers. The clean single-$n/(o-3/2)$ form is exactly the special case
$\delta\in\{0,1\}$.

### Where the two exponents come from, and why the interior optimum is $\delta_j n/(o - 1/2 - \delta_j)$

The objective's exponent $o - 1/2$ and the interior optimum's denominator
$o - 1/2 - \delta_j$ look suspiciously alike; they should, because the second is the first minus
$\delta_j$, and both come from one place.

**The objective exponent $o - 1/2$.** To cover a fixed span of simulated time,

$$
\#\mathrm{batches} = \frac{T}{\Delta t_{\mathrm{batch}}},
$$

and one batch advances

$$
\Delta t_{\mathrm{batch}}
= \frac{\mathbb{E}[\ell]}{P_{\mathrm{total}}}
= \frac{\mathbb{E}[\ell]}{k_{\max}\binom{N}{o}}.
$$

With $\mathbb{E}[\ell]=c\sqrt{N}$ and $\binom{N}{o}\approx N^o/o!$,

$$
\#\mathrm{batches}
\propto
\frac{k_{\max}\binom{N}{o}}{\mathbb{E}[\ell]}
\propto
\frac{k_{\max}N^o}{\sqrt{N}}
=
k_{\max}N^{\,o - 1/2}.
$$

So the exponent splits cleanly: the $o$ is the reactant-set combinatorics
$\binom{N}{o}\sim N^o$ (the clock), and the $-1/2$ is the collision length
$\mathbb{E}[\ell]\sim\sqrt{N}$ sitting in the denominator. Neither half has anything to do with a
particular reaction.

**The filler-dilution exponent $\delta_j$.** On the branch where reaction $j$ sets $k_{\max}$,

$$
k_{\max}
= \frac{A_j}{\operatorname{ff}(k_0,\delta_j)}
\approx
\frac{A_j}{k_0^{\delta_j}}.
$$

Reaction $j$ carries $\delta_j$ fillers, and its adjusted rate is diluted by the $\mathbf{K}$ falling
factorial. So on that branch the objective is

$$
f(k_0)
=
\frac{A_j(n+k_0)^{\,o - 1/2}}{k_0^{\delta_j}}.
$$

**The optimum.** This is an instance of the elementary fact

$$
\min_{k>0}\ \frac{(n+k)^p}{k^q}
\quad\Longrightarrow\quad
k^\star = \frac{qn}{p-q}
\qquad
\text{for } p > q > 0.
$$

The first-order condition equates the numerator's logarithmic growth rate $p/(n+k)$ with the
denominator's $q/k$. With $p=o-1/2$ (the batch-cost exponent) and $q=\delta_j$ (the filler dilution),

$$
k_0^\star
=
\frac{\delta_j n}{(o - 1/2)-\delta_j}.
$$

So $o - 1/2 - \delta_j$ is exactly $p-q$: the *net* power of $k_0$ in the objective after the
numerator's growth and the denominator's dilution are combined. It must be positive
($o - 1/2 > \delta_j$) for a finite optimum to exist. Otherwise dilution always wins, $f$ falls
monotonically, and you would add filler without bound. This degenerate case is avoided by only padding
up to order $o$ and taking $o\ge 2$.

**Order-2 collapse.** The only padded reactions have $\delta=1$, so

$$
k_{\max}
=
\max\left(\frac{C_{\mathrm{padded}}}{k_0}, C_{\mathrm{flat}}\right),
\qquad
C_{\mathrm{padded}}=\max_{\mathrm{order}\ 1} A_j.
$$

The single interior optimum is

$$
k_0
=
\frac{1\cdot n}{2 - 1/2 - 1}
=
\frac{n}{1/2}
=
2n,
$$

equivalently $2n/(2o-3)$ at $o=2$. The single crossover is
$C_{\mathrm{padded}}/C_{\mathrm{flat}}$, giving

$$
k_0^\star = \min(2n,\mathrm{crossover}).
$$

### Worked order-3 example

A concrete $o=3$ CRN with one reaction of each order and volume $v=10^6$:

$$
\begin{array}{rcll}
A &\to& 2D & \mathrm{rate}=1,\quad \mathrm{order}=1,\quad \delta=2,\\
A+B &\to& 3D & \mathrm{rate}=10,\quad \mathrm{order}=2,\quad \delta=1,\\
A+B+C &\to& 4D & \mathrm{rate}=1,\quad \mathrm{order}=3,\quad \delta=0.
\end{array}
$$

The last reaction is the flat floor. The adjusted-rate numerators are

$$
A_a =
\frac{\mathrm{rate}}{v^{a-1}}\cdot\mathrm{symmetry},
$$

where the symmetry is that of the *padded* reactant multiset. The order-1 reaction's two identical
fillers therefore contribute a factor of $2!$:

$$
A_1 = 1\cdot 1\cdot 2! = 2,
\qquad
A_2 = \frac{10}{v}\cdot 1\cdot 1! = 10^{-5},
\qquad
A_3 = \frac{1}{v^2}\cdot 1\cdot 0! = 10^{-12}.
$$

Hence

$$
k_{\max}(k_0)
=
\max\left(\frac{A_1}{k_0^2},\frac{A_2}{k_0},A_3\right),
$$

with **two** crossovers:

$$
x_{21} = \frac{A_1}{A_2} = 2\cdot 10^5
\qquad
(\delta=2 \to \delta=1),
$$

$$
x_{10} = \frac{A_2}{A_3} = 10^7
\qquad
(\delta=1 \to \delta=0).
$$

The two padded branches have interior optima

$$
4n
\qquad
\left(\delta=2:\ \frac{2n}{o - 1/2 - 2} = \frac{2n}{1/2}\right)
$$

and

$$
\frac{2n}{3}
\qquad
\left(\delta=1:\ \frac{n}{o - 1/2 - 1} = \frac{n}{3/2}\right).
$$

The optimum then walks through four regimes as $n$ grows. The measured
$\arg\max_K \sqrt{N}p$, which minimizes $k_{\max}N^{\,o-1/2}$, matches the prediction:

| $n$ | measured $K^\star$ | optimum | which branch |
|---:|---:|---:|---|
| $10^4$ | $40{,}567$ | $4n$ | order-1 interior ($\delta=2$) |
| $5\cdot 10^4,\ 10^5$ | $183{,}384$ (both) | $x_{21}=2\cdot 10^5$ | $\delta=2\to\delta=1$ crossover (plateau) |
| $10^6$ | $630{,}116$ | $2n/3$ | order-2 interior ($\delta=1$) |
| $10^7$ | $6{,}485{,}995$ | $2n/3$ | order-2 interior ($\delta=1$) |
| $3\cdot 10^7,\ 10^8$ | $9{,}787{,}303$ (both) | $x_{10}=10^7$ | $\delta=1\to\delta=0$ crossover (plateau) |

So $k_0^\star(n)$ is piecewise

$$
4n
\to
x_{21}
\to
\frac{2n}{3}
\to
x_{10}.
$$

That is, two $n$-dependent segments with *different* interior formulas, joined by two
config-independent crossover plateaus -- not a single $n/(o-3/2)$ under a single crossover. This CRN
is in the benchmark notebook as "Order-3 example".

### What `k_reset_target` actually computes

`k_reset_target` implements the $\delta=1$ case:

$$
\min\left(\frac{n}{o-3/2},\ \mathrm{crossover}\right),
$$

where `crossover` is the $\delta=1$-to-flat handoff
$A_{\mathrm{padded}}/C_{\mathrm{flat}}$ (`crossover_k_count`). That is **exact for $o=2$**: there
$\delta\in\{0,1\}$ is all there is, with one padded branch, one crossover, and interior optimum $2n$.
For $o\ge 3$ it is an approximation. In the example above it returns

$$
\min\left(\frac{2n}{3}, 10^7\right),
$$

which is correct on the $2n/3$ segment and its $x_{10}$ plateau, but blind to the small-$n$
$4n/x_{21}$ regime, where a $\delta=2$ branch sets $k_{\max}$. The full candidate search above,
including per-branch interiors $\delta_j n/(o-1/2-\delta_j)$ and all crossovers, would be needed
there. No benchmark CRN is affected; all are order $2$.

## Comparison to the old policy ($K=n$)

The old heuristic reset $\mathbf{K}$ to $n$ whenever $\mathbf{K}$ drifted out of
$[n/2,2n]$. Relative to

$$
k_0^\star = \min(2n,\mathrm{crossover}),
$$

the behavior is:

- when $n \gg \mathrm{crossover}$ (LV at its population peaks), $K=n$ *overshoots* the optimum,
  inflating $N$ and wasting batches. This is where the new policy wins.
- it also makes the non-passive fraction wobble and jump, because that fraction depends on the
  drifting, periodically reset $\mathbf{K}$. This is the artifact behind the "the fraction differs at
  near-identical configurations" observation. Pinning $\mathbf{K}$ at the constant crossover removes
  the drift and the jumps.

## Empirical validation

All measured with the actual sampled $\mathbb{E}[\ell]$ (`sample_collision`) and exact $p$. At a frozen
config, $\arg\max_K \mathbb{E}[\ell] \cdot p$ is the batch-optimal $\mathbf{K}$.

- **Batch counts (pure batch, $n=10^5$):** LV has **$1.4$-$1.5\times$ fewer batches** under
  $\min(2n,\mathrm{crossover})$ than under $K=n$, with a flat non-passive fraction of about $0.62$
  versus the old policy's $0.28$-$0.63$ with reset jumps. Oregonator is about $1.03\times$ because its
  crossover is about $n$, so $K=n$ was already near-optimal.
- **Volume scaling ($n$ fixed at $10^5$, $v$ varied):** the optimum doubles when $v$ doubles and halves
  when $v$ halves, confirming $k_0^\star\propto v$, not $n$. The crossover is $1.00v$ for LV and
  $0.52v$ for Oregonator across $v\in\{n/2,n,2n\}$.
- **Rössler, both regimes:** $\arg\max_K \mathbb{E}[\ell] \cdot p = 186\mathrm{k}\approx 2n$ at the IC
  ($n=10^5$, so $2n<\mathrm{crossover}=3\cdot 10^6$), and
  $3.13\cdot 10^6\approx\mathrm{crossover}$ at an evolved config
  ($n=6.1\cdot 10^6$, so $\mathrm{crossover}<2n$). The formula picks the right branch in each case.

## Implementation notes

- The target is computed in `BatchSimulator::k_reset_target` (`src/simulator_crn.rs`), and
  `reset_k_count` moves $\mathbf{K}$ toward it. It is recomputed and checked before **every** batch.
  This is cheap: the crossover is config-independent and cached at construction in `crossover_k0`.
  The band rule in `run()` rebuilds the transition arrays only when $\mathbf{K}$ has drifted more than
  a factor `K_RESET_BAND_FACTOR` ($1.1$) from the target, so $\mathbf{K}$ tracks the $2n$ branch to
  within about $10\%$ without rebuilding on every reaction. Once it reaches the config-independent
  crossover, it stops firing.
- **Always dynamic.** Whether a run's population moves enough to need re-tuning $\mathbf{K}$ is
  undecidable in general, because CRNs can simulate Turing machines. There is no separate "static vs
  dynamic" mode: the target is always recomputed and the band always applies. For crossover-binding
  CRNs (LV, Oregonator) the target is constant, so the band fires once and never again. For CRNs whose
  population moves, it tracks $2n$ (Rössler upward past the crossover; the Shrinking CRN
  $A\to\emptyset,\ B\to\emptyset,\ 2A\to 2B$ downward, where a frozen crossover-sized $\mathbf{K}$
  would be about $7.5\times$ slower).
- **The non-passive-fraction "jumps"** seen when plotting are these rebuilds: between resets
  $\mathbf{K}$ is fixed while $n$ moves, so the fraction drifts, then snaps back to the optimum at each
  reset. A smaller `K_RESET_BAND_FACTOR` makes the jumps smaller and more frequent; removing them
  entirely would require rebuilding every batch.
- `k0_manual_multiplier > 0` overrides the target with `round(mult*n)` for $\mathbf{K}$ sweeps
  (test-only).
