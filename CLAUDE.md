# CLAUDE.md

**At the start of every session, read [`CONTRIBUTING.md`](CONTRIBUTING.md) before doing anything
else.** It documents the build/toolchain gotchas (nightly Rust pin, Dropbox/`CARGO_TARGET_DIR`,
Windows file locks, testing CI builds locally) and the PyPI release process; this file only
summarizes some of them.

## What this project is

`batss` is a research library for **fast stochastic simulation of chemical reaction networks (CRNs)
and population protocols** — mathematical models of interacting-particle dynamics studied in
molecular computing, distributed computing, and nonlinear dynamics. It is developed by the UC Davis
molecular computing group (repo: `github.com/UC-Davis-molecular-computing/batss`) and was formerly
named `ppsim`.

The point of the library is *performance*: it simulates the continuous-time Markov chain of a CRN
using a batching algorithm (Berenbrink-style; see arXiv:2508.04079) that advances ~√n reactions at a
time, so it can reach large population sizes (n up to ~1e10) far faster than a naive Gillespie SSA.
It falls back to exact Gillespie (via the `rebop` crate) when that is faster. The example CRNs used
throughout — Dimerization, Lotka–Volterra, Rössler–Willamowski, and the Oregonator (a model of the
Belousov–Zhabotinsky oscillating reaction) — are standard textbook dynamical-systems / chemical-
kinetics models used purely as simulation benchmarks.

## Architecture

- **Rust core** in [`src/`](src/): the simulator (`simulator_crn.rs`), the sampling urn
  (`urn.rs`), numerics (`util.rs`), exposed to Python via **pyo3 / maturin** as the `batss_rust`
  extension. `BatchSimulator` alternates between the batching engine and rebop's Gillespie;
  the batch↔Gillespie switching logic is in `SwitchState` / `run()` (see
  [`GILLESPIE_SWITCH_LOGIC.md`](GILLESPIE_SWITCH_LOGIC.md)).
- **Python API** in [`python/batss/`](python/batss/): `batss.Simulation`, `batss.species("A B")`,
  reactions built with the `>>` / `|` operators and `.k(rate)` (e.g. `(a + b >> 2*b).k(1.5)`),
  `sim.run(end_time, history_interval)`, `sim.history` (a pandas DataFrame). `convert_to_uniform` pads
  the CRN with the filler species `K`/`W` to make every reaction uniform; these are simulated but hidden
  from `history` / `config_dict` (see `Simulation._visible_indices`).
- **Benchmarks** in [`benchmark/`](benchmark/): `dimerization_benchmarks.py` (`oregonator_spec()`,
  `dimerization_spec()`, `mode_split()`), `generate_gallery_figures.py` (the authoritative definitions of
  the 5 benchmark CRNs), `benchmark.ipynb`, and cached runtime JSON in `benchmark/data/` (committed).
  The reusable `CRNSpec` / `benchmark_runtimes` / `plot_runtimes` / `plot_trajectory` interface lives in
  [`python/batss/benchmarking.py`](python/batss/benchmarking.py) — **not public API**; keep it out of the
  user-facing docs.
- **Docs**: `README.md` is **generated** from `README.ipynb` — never edit it by hand; run
  `python scripts/make_readme.py` (see [`CONTRIBUTING.md`](CONTRIBUTING.md)). Further usage examples are
  in [`examples/`](examples/).

## Key commands

Rust must be rebuilt after any change to `src/`:

```
CARGO_TARGET_DIR="$LOCALAPPDATA/batss-cargo-target" maturin develop --release
```

- The `CARGO_TARGET_DIR` outside the Dropbox-synced tree avoids an intermittent Windows file-lock
  (`os error 32`) when writing `target/`.
- If the copy of the built `.pyd` fails with "used by another process", a **Jupyter kernel with
  `import batss` loaded is holding it** — close the notebook/kernel (or rename the old `.pyd` aside)
  and re-run.
- Fast type-check without building the extension: `cargo check --release`.

Tests: `python -m unittest tests.batss_tests` (plain `unittest`, no pytest).

## Conventions

- Rate constants are given in the deterministic/macroscopic convention; the simulator divides a
  k-reactant reaction's rate by `volume**(k-1)`, with `volume` defaulting to the total initial
  count. `benchmarking._batss_sim` passes `volume=n` explicitly so initial conditions need not sum
  to `n` (e.g. an initial condition placed on a limit cycle).
- Rust: run `cargo check --release` before building. Keep the hot `run()` loop allocation-free.
- See `CHANGES_2026-07-03.md` for the Gillespie-mode urn-sync fix and `GILLESPIE_SWITCH_LOGIC.md`
  for the mode-switching heuristic and its open questions.
