# Contributing to the batss project

## Regenerating README.md

**`README.md` is generated from a Jupyter notebook. Never edit it by hand; your changes will be overwritten. Instead edit the notebook; details below.**

It is the `nbconvert` output of [`README.ipynb`](README.ipynb), so that every example in the README is
one a reader can actually download and run, and so the two cannot drift apart (they had, badly, before
this was automated). The workflow is:

1. Edit `README.ipynb` and **run it yourself in Jupyter**, top to bottom.
2. Look at the output. When it looks the way you want it:

```
python scripts/make_readme.py
```

3. Commit `README.ipynb`, `README.md`, and the regenerated `README_files/*.png` together.

**The script does not run the notebook** — deliberately. The simulator is not reproducible even with a
fixed seed ([#14](https://github.com/UC-Davis-molecular-computing/batss/issues/14)), so re-executing it
here would throw away the output you had just checked over and replace it with a different sample. Running
the notebook is your step; converting it is the script's. With no notebook execution and no GIF rebuild,
the script is byte-for-byte idempotent: running it twice changes nothing.

Three things it handles that are easy to get wrong by hand:

- **Images must be served from `raw.githubusercontent.com`.** `pyproject.toml` sets
  `readme = "README.md"`, so the file becomes the PyPI long description, and PyPI cannot resolve
  repo-relative paths. A `github.com/.../blob/...` URL returns an HTML page rather than the image, so it
  will silently fail to render on PyPI.
- **Progress bars and `ipywidgets` sliders are stripped**, since `nbconvert` renders them as a stale text
  repr (a progress bar frozen at `0%|`, or `interactive(children=(IntSlider(...`) that is pure noise in a
  static README.
- **The `<style>` block that pandas emits with every DataFrame is stripped.** GitHub's sanitizer removes
  the `<style>` element but then renders its CSS *text* as visible prose, so the raw rules would appear
  above every table. The `<table>` renders fine without it.

### The animated GIFs

A slider cannot survive `nbconvert`, and GitHub cannot render a live widget inside a notebook either, so
where `README.ipynb` builds a slider the README embeds a GIF of that slider being dragged. These are
rendered offscreen with matplotlib, not screen-recorded:

```
python scripts/make_readme.py --gifs
```

`--gifs` is **off by default**, because building them re-runs the simulations — so, per #14 above, they
come out different every time, and rebuilding on every conversion would rewrite a couple of hundred KB of
binary even when nothing had changed. Pass it only when you have actually changed the slider cells. It adds
about 12 seconds.

If the notebook is newer than the GIFs, the script says so, so you get a nudge rather than silent staleness.

### Other notebooks

`examples/` and `benchmark/` are not converted to markdown; GitHub renders them directly. Re-execute with:

```
python -m jupyter nbconvert --to notebook --execute --inplace examples/crn_examples.ipynb
```

## Compiling the Rust code

The Python package `batss` is a thin Python layer (in the `python/` directory) wrapping a Rust extension module compiled from the code in `src/`. We use [maturin](https://www.maturin.rs/) to compile the Rust code and bind it to Python. maturin is declared as the build backend in `pyproject.toml`, so you do not need to install it separately if you install the package with `pip`; but for local development it is convenient to install it explicitly:

```
pip install maturin
```

You will also need a [Rust toolchain](https://rustup.rs/) (`cargo` and `rustc`).

> **Nightly Rust is required.** The repo pins the nightly toolchain via `rust-toolchain.toml`,
> because `src/util.rs` uses the nightly-only `f128` type. You don't need to do anything for
> this: rustup reads the pin automatically and downloads nightly the first time you build in
> this repo. What you must **not** do is override it — setting `RUSTUP_TOOLCHAIN=stable`,
> `rustup override set stable`, or `cargo +stable` will bypass the pin and fail with a wall of
> confusing errors in `util.rs` (`f128`, `cast_signed`, error E0554 "#![feature] may not be used
> on the stable release channel"). If you see those errors, you are on stable; nothing is wrong
> with the code. The GitHub CI honors the pin as well.

> **Dropbox / synced-folder note:** if your clone lives inside Dropbox (or another file-syncing
> folder), point cargo's build directory outside the synced tree, e.g.
> `$env:CARGO_TARGET_DIR = "$env:LOCALAPPDATA/batss-cargo-target"` (PowerShell) or
> `export CARGO_TARGET_DIR="$HOME/batss-cargo-target"` (macOS/Linux/WSL). This avoids syncing
> gigabytes of build artifacts and an intermittent Windows file-lock error (`os error 32`) when
> cargo writes `target/`. Note that with this set, `maturin build` writes wheels to
> `$CARGO_TARGET_DIR/wheels/` instead of `target/wheels/` (unless you pass `--out`).

Both of the commands below build the Rust extension and, by default, install it into the **currently active** virtual environment, so activate the environment you want to use first, e.g.:

```
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### Development mode

For iterating locally, use `maturin develop`. It compiles the Rust code and installs the extension module in-place into the active virtual environment, so that `import batss` picks up your changes without building or installing a wheel:

```
maturin develop
```

By default this produces an **unoptimized (debug) build**, which compiles quickly but runs much slower. Because `batss` is a performance-oriented simulation package, when you want to test with realistic speed, build in release mode:

```
maturin develop --release
```

Re-run the appropriate command after each change to the Rust code (changes to the Python code in `python/` take effect immediately without recompiling).

> **Windows note — "file is being used by another process" (`os error 32`):** if `maturin develop` fails with an error like
> ```
> 💥 maturin failed
>   Caused by: Failed to copy ...\target\release\batss_rust.dll to ...\batss_rust.cp313-win_amd64.pyd:
>   The process cannot access the file because it is being used by another process. (os error 32)
> ```
> it means a running process still has the compiled extension loaded, and Windows locks the file for the lifetime of any process that has imported it. The usual culprit is a **Jupyter kernel or Python REPL that has run `import batss`** (for example, an open notebook in VS Code or JupyterLab). **Restart or shut down that kernel** (VS Code: the **Restart** button in the notebook toolbar; JupyterLab: *Kernel → Shut Down Kernel*), then re-run `maturin develop`. As a habit, restart the kernel before rebuilding whenever you have imported `batss`. If you can't find the offending process, you can identify and kill it from PowerShell:
> ```powershell
> # find which python process has the extension loaded
> Get-Process python* | Where-Object { $_.Modules.FileName -like '*batss_rust*' } | Select-Object Id, Path
> # then stop it (this discards that process's in-memory state)
> Stop-Process -Id <PID> -Force
> ```
> (Linux and macOS do not have this problem, since they allow overwriting a loaded shared library.)

### Production mode

To produce an optimized, distributable wheel (the same kind uploaded to PyPI, but built for your current platform and Python version), use

```
maturin build -r
```

The resulting `.whl` file is written to `target/wheels/`. You can then install it into any environment with, e.g.:

```
pip install target/wheels/batss-1.0.3-<platform-tags>.whl
```

(replace the filename with the actual one produced).

Note that for actually publishing to PyPI you do not build wheels manually; that is handled automatically by the GitHub action described in the next section.

### Testing the CI build locally

The GitHub action (next section) uses [`PyO3/maturin-action`](https://github.com/PyO3/maturin-action), the official GitHub Action wrapper around the same maturin tool used above. On every platform it effectively runs

```
maturin build --release --out dist --find-interpreter
```

plus `maturin sdist --out dist` in a separate job. To test locally whether a platform's wheel will build, run exactly that command on that platform. Differences from real CI to be aware of:

- **Linux wheels are built inside manylinux Docker containers** (for compatibility with old glibc). Building on a regular Ubuntu machine or WSL still verifies that the code compiles and links on Linux, which is the part that can realistically break — just not the manylinux glibc compliance.
- `--find-interpreter` builds one wheel per Python interpreter it can find. On CI, `setup-python` guarantees discovery; locally it can fail with "Could not find any interpreters" (e.g., in non-interactive shells, or when your only Python is a venv). If so, point maturin at a specific interpreter instead: `maturin build --release --out dist -i path/to/python`.
- `rebop` is a **required dependency on all platforms**, including macOS, where it now builds cleanly. batss uses only rebop's pure-Rust exact-Gillespie engine (the fallback in `simulator_crn.rs`), not rebop's own Python bindings, so it is declared `rebop = { version = "0.9.7", default-features = false }` to keep rebop's optional `pyo3` binding **off** — see the next section for why that matters. (History, so nobody reintroduces the old workaround: macOS builds failed with rebop in Aug 2025, so it was feature-gated off there via `--no-default-features`, then removed entirely in v1.0.2; it was re-added for v1.0.3, which building without is no longer possible. The Aug-2025 failure was never definitively diagnosed, but does not reproduce with the current pyo3 0.28 / rebop 0.9.7 stack.)

### Python version support, and the pyo3 / numpy / rebop version coupling

The Rust extension binds to CPython through **pyo3**, and each pyo3 release supports a fixed range of CPython versions. **This is what determines which Python versions batss can build against.** As of this writing batss uses **pyo3 0.28**, which supports **CPython 3.7–3.14**. If you build against a Python *newer* than pyo3's maximum, the build fails early in pyo3's build script with:

```
error: the configured Python interpreter version (3.NN) is newer than PyO3's maximum supported version (3.MM)
```

This is not a bug in batss and not specific to any platform — it just means pyo3 is too old for that interpreter.

**Three Rust dependencies are version-coupled and must move together** (in `Cargo.toml`):

- **`pyo3`** — the binding itself; sets the CPython support ceiling.
- **`numpy`** (the [rust-numpy](https://github.com/PyO3/rust-numpy) crate) — its minor version tracks pyo3's one-to-one (`numpy 0.28` ⇄ `pyo3 0.28`). Keep them on the same minor.
- **`rebop`** — also binds pyo3, but only through an optional feature. Because batss uses rebop's pure-Rust `gillespie` API, it sets `default-features = false` to keep rebop's `pyo3` binding disabled. If it were enabled, rebop would pull a second pyo3 into the extension, which cannot link. (rebop ≤ 0.9.2 pinned pyo3 as a *required* dependency, which transitively froze batss's pyo3 version and, before mid-2026, capped batss at Python 3.13; rebop ≥ 0.9.7 made pyo3 optional, which is what lets batss pick its own.)

Only **one** pyo3 version may be linked into the extension module, so all three must agree. Verify the resolved graph with:

```
cargo tree -i pyo3     # expect a single pyo3 version, contributed by batss + numpy only (not rebop)
```

**To support a newer CPython** (e.g. when a new 3.x ships): bump `pyo3` and `numpy` together to a version whose supported range includes it (see the [pyo3 changelog](https://pyo3.rs/main/changelog)), make sure `rebop` is recent enough to compile against that pyo3, run `cargo update`, then rebuild and run the tests. A pyo3 major bump can require small code changes — consult the [pyo3 migration guide](https://pyo3.rs/main/migration). The 0.24 → 0.28 bump done for v1.0.3, for instance, needed only `Python::with_gil(...)` → `Python::attach(...)` (two call sites) and an explicit `#[pyclass(from_py_object)]` on `SwitchState`.

> **Why this bites at *release* time — the CI `python-version: "3.x"` trap.** The build workflow uses `actions/setup-python` with `python-version: 3.x`, which resolves to the *latest stable* CPython on the runner — and that advances over time. Combined with `maturin build --find-interpreter` (which builds a wheel for **every** interpreter it finds), a runner Python newer than pyo3 supports fails the whole release job on **every** platform, even though nothing in batss changed. This is exactly what threatened v1.0.3: v1.0.0–1.0.2 were released in Aug 2025, before CPython 3.14; by mid-2026 `3.x` resolved to 3.14, which pyo3 0.24 could not build. The fix was moving to pyo3 0.28. Keep pyo3/numpy current (or pin the CI Python) so a new CPython release doesn't silently break publishing.

### Troubleshooting a broken dev environment

Symptoms below are all things that have actually bitten this project, typically on a machine that
has sat idle while development happened elsewhere. Each is *environment* breakage — the code is
fine, so don't go looking for a bug in `src/`.

> **`maturin develop` succeeds but your Rust changes have no effect** — and Python raises
> `AttributeError: 'builtins.BatchSimulator' object has no attribute '<name>'` for a method that
> plainly exists in `src/simulator_crn.rs` inside the `#[pymethods]` block and is declared in
> `batss_rust.pyi`. **This is the nastiest failure in this list, because nothing reports an
> error.** maturin builds an **abi3** wheel (`batss-*-cp310-abi3-win_amd64.whl`), which installs
> as *untagged* `batss_rust.pyd`. But CPython's extension finder checks *version-tagged* names
> first, so a leftover `batss_rust.cp313-win_amd64.pyd` from a pre-abi3 build silently shadows
> every rebuild you do. Confirm which file is actually loaded:
>
> ```
> python -c "from batss.batss_rust import batss_rust as m; print(m.__file__)"
> ```
>
> If that prints anything other than plain `batss_rust.pyd`, rename the tagged file aside (the
> convention in `python/batss/batss_rust/` is a `.old.pyd` suffix; `*.pyd` is gitignored there, so
> this is a local-only change) and re-run. Only `batss_rust.pyd` should remain active.

> **`did not find executable at '...\uv\python\cpython-X.Y.Z-windows-x86_64-none\python.exe'`,**
> followed by pyo3's `error: Python script failed`. The venv's base interpreter is gone: uv prunes
> old patch releases from its managed-Python store, but `.venv/pyvenv.cfg`'s `home =` line still
> points at the pruned one, and `.venv/Scripts/python.exe` is a uv *trampoline* that resolves
> through it. Everything Python-facing dies at once — tests, Jupyter, `maturin develop` — and
> because `.vscode/settings.json` passes that interpreter to pyo3 as `PYO3_PYTHON`, so does
> `cargo check`, which is why **rust-analyzer goes completely dark and reports it as a Rust
> error**. Read the exact version out of `.venv/pyvenv.cfg` and reinstall precisely that one:
>
> ```
> uv self update                 # an old uv's manifest may not list the version you need
> uv python install 3.13.13      # <- the exact version_info from pyvenv.cfg
> ```
>
> This restores the `home` path verbatim, so the venv and its installed packages keep working
> untouched. Only recreate the venv (`uv venv --python 3.13`, reinstall, rebuild) if that exact
> patch release is genuinely unavailable — a *different* minor version cannot back the existing
> `site-packages`, whose compiled extensions carry a `cp3NN` ABI tag.

> **`maturin failed: Failed to find pip`.** The venv was created by uv and has no `pip`. Use
> `maturin develop --release --uv`, as maturin's own error message suggests.

> **VS Code warns "your workspace uses a rust-toolchain file with a toolchain too old for the
> extension shipped rust-analyzer".** The pin in `rust-toolchain.toml` is the floating `nightly`
> channel, so a machine that hasn't built in a long time can be a year behind the rust-analyzer
> that VS Code ships. `rust-toolchain.toml` requests `components = ["rust-analyzer"]` so rustup
> supplies a *version-matched* analyzer; keep the toolchain current so that analyzer is recent too:
>
> ```
> rustup update nightly
> rustup component list --toolchain nightly | grep rust-analyzer   # expect "(installed)"
> ```

> **rust-analyzer panics with "`Python<'^0>: Copy` has escaping bound vars, so it cannot be
> wrapped in a dummy binder",** and `textDocument/diagnostic` fails for `src/simulator_crn.rs`.
> **This is an upstream rust-analyzer bug, not a problem with the code** — a crash in the
> next-generation trait solver it vendors from rustc, hit while borrow-checking the higher-ranked
> `Python::attach(|py| ...)` closures that pyo3 requires (`Python<'py>` carries a bound lifetime,
> and the solver asserts when asked to prove `Python<'^0>: Copy` before that lifetime is
> instantiated). It is **not** fixed by matching the analyzer to the toolchain — both the bundled
> and the rustup-supplied analyzer panic identically — and no setting disables the new solver.
> The workaround in `.vscode/settings.json` is:
>
> ```jsonc
> "rust-analyzer.diagnostics.enable": false
> ```
>
> That switch turns off only rust-analyzer's *native* diagnostics. `cargo check` (flycheck) still
> reports real rustc errors and warnings, and completion, hover, goto-definition and inlay hints
> are unaffected — so in practice you lose little. Worth retrying after a rust-analyzer update; if
> the panic is gone, delete the setting. A narrower alternative that may or may not work,
> depending on whether the analyzer still runs borrowck before filtering, is
> `"rust-analyzer.diagnostics.disabled": ["need-mut", "unused-mut"]`.

> **`error: failed to remove ...\target\debug\deps\*.rcgu.o: ... used by another process
> (os error 32)` coming from rust-analyzer** (not from your own `cargo` runs). rust-analyzer runs
> its own `cargo check` and, by default, writes to `target/` inside the repo — which is the
> synced-folder hazard described above. `.vscode/settings.json` sets `CARGO_TARGET_DIR` under
> `rust-analyzer.cargo.extraEnv` to move that output out of the synced tree. Two caveats: the path
> there is **absolute and machine-specific**, so adjust it on a new machine; and it is
> deliberately *not* the same directory used for `maturin develop`, since sharing one would make
> flycheck and your builds block on the same cargo lock.

## Deploying to PyPI

Releasing is driven by the [GitHub CLI](https://cli.github.com/) (`gh`) plus the helper scripts in [`scripts/`](scripts/) — a Bash version (`.sh`, for macOS/Linux/WSL/Git Bash) and a PowerShell version (`.ps1`, for Windows) of each. The manual web-UI steps remain documented under "Doing it by hand" below.

### Install and authenticate `gh` (one time)

```
# macOS
brew install gh
# Debian/Ubuntu  (other distros: https://github.com/cli/cli/blob/trunk/docs/install_linux.md)
sudo apt install gh
# Windows
winget install --id GitHub.cli        # or: choco install gh   /   scoop install gh

gh auth login       # choose GitHub.com + HTTPS
```

Verify with `gh auth status`. Dispatching a workflow (`gh workflow run`) needs the **workflow** token scope; if a dispatch is rejected with a permissions error, run `gh auth refresh -s workflow`.

### Test that every platform builds — without publishing

```
scripts/test-build.sh            # bash: dispatch the workflow on main (build-only) and watch it
scripts/test-build.sh <branch>   # bash: test a different branch
./scripts/test-build.ps1         # Windows PowerShell equivalent
./scripts/test-build.ps1 <branch>
```

`test-build.sh` fires the workflow via `workflow_dispatch`. Because the publish job is gated on `if: github.event_name == 'release'`, this builds Linux/musllinux/Windows/macOS + sdist and **uploads nothing to PyPI** — no release, no tag. Run it before every release; a platform break is then caught with nothing to clean up.

### Cut a release and publish to PyPI

```
scripts/release.sh               # bash
./scripts/release.ps1            # Windows PowerShell equivalent
```

`release.sh` reads the version from `Cargo.toml` and refuses to proceed unless (a) that version is strictly newer than the latest GitHub release/tag, (b) the tag `vX.Y.Z` does not already exist, and (c) your local `main` matches `origin/main` (so CI builds exactly what you tested). It then asks you to type the tag to confirm, creates the GitHub Release (which triggers the build **and** the PyPI publish), and watches the run. **A published PyPI version cannot be re-uploaded**, so bump `Cargo.toml` first (step 1 under "Doing it by hand").

### Doing it by hand (web UI)

Deploying to the [PyPI website](https://pypi.org/project/batss/) is how users are able to install via `pip install batss`. This is done by a GitHub action in .github/workflows/build_and_publish.yml. It builds binary wheels (so that users do not need a Rust compiler to install) for each major platform and Python version and uploads them to PyPI. This is done automatically whenever there is a new GitHub release. The steps are

1. Bump the version number **in `Cargo.toml` — the only place the version is stored**, e.g., change 1.0.0 to 1.0.1 if the last uploaded version was 1.0.0 (or bump minor or major version numbers if appropriate according to [semantic versioning](https://semver.org/)). For the rest of this section assume the version number is 1.0.1. Then run `cargo update --workspace` to propagate it into `Cargo.lock` (any cargo build also does this automatically) and commit both files. Everything else derives from `Cargo.toml` automatically: `pyproject.toml` declares `dynamic = ["version"]`, [maturin's officially supported way](https://www.maturin.rs/metadata) of using the Rust crate version as the Python package version; `uv.lock` does not record a version for batss (it defers to the dynamic version); `batss.__version__` reads the installed package metadata at runtime; and `doc/conf.py` parses `Cargo.toml` directly.

2. Commit this and other changes to the main branch and push to github.

3. On the [Github page](https://github.com/UC-Davis-molecular-computing/batss), click on Releases-->Create a new release. Title the release `v1.0.1` (or whatever is the version number). This can actually be anything but that is a good title. More importantly, press `Tag: Select tag` and type `v1.0.1`; this must be exactly that, i.e., it must be the lowercase letter `v` followed by the version number in pyproject.toml. Press Enter (i.e., don't just click outside once you've typed; you have to press Enter for it to create the tag). Importantly, there must be no tag already called `v1.0.1`; to double-check, click on Tags at the top of the releases page to see existing tags. If that tag exists, delete it. 

4. Click Publish release.

5. Click on Actions at the top of the page to ensure that the build_and_publish.yml action is running. Because it must compile Rust files for many different platforms (and for each many different Python versions), this takes several minutes. If you are testing out something repeatedly in a short time, it is best to delete all or most of the parts of the Github action that deal with building, since that will eat up limited computation time alloted by Github.

6. If the action successfully completes, go to https://pypi.org/project/batss/ to verify that it is the latest version, 1.0.1 in this case.

7. Recall above that the tag must not already be present. If you need to redo this due to a mistake, not only must you delete the release first, you must also click on Tags at the top of the Releases page and delete the tag. If you do not, then it will not successfully upload to PyPI.

> **Dry-run tip (safe all-platform build test):** the workflow also has a manual trigger (Actions → "Build and Publish Python Package" → Run workflow, or `gh workflow run build_and_publish.yml --ref main`). The publish job is gated with `if: github.event_name == 'release'`, so a manual `workflow_dispatch` builds **every** platform (Linux/musl/Windows/macOS + sdist) **without uploading to PyPI**. Use it to confirm all platforms build *before* creating the release; if a platform fails there is no release or tag to clean up. Publishing to PyPI happens **only** when you create an actual GitHub Release (steps 3–4 above).