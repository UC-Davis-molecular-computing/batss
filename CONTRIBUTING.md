# Contributing to the batss project

## Compiling the Rust code

The Python package `batss` is a thin Python layer (in the `python/` directory) wrapping a Rust extension module compiled from the code in `src/`. We use [maturin](https://www.maturin.rs/) to compile the Rust code and bind it to Python. maturin is declared as the build backend in `pyproject.toml`, so you do not need to install it separately if you install the package with `pip`; but for local development it is convenient to install it explicitly:

```
pip install maturin
```

You will also need a [Rust toolchain](https://rustup.rs/) (`cargo` and `rustc`).

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

> **macOS note:** the `rebop` dependency does not build on macOS, so it is an optional Cargo feature. On macOS, disable the default features:
> ```
> maturin develop --no-default-features
> ```
> (This mirrors what the CI does for macOS in `.github/workflows/build_and_publish.yml`.)

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

To produce an optimized, distributable wheel (the same kind uploaded to PyPI, but built for your current platform and Python version), use `maturin build --release`:

```
maturin build --release
```

The resulting `.whl` file is written to `target/wheels/`. You can then install it into any environment with, e.g.:

```
pip install target/wheels/batss-1.0.2-<platform-tags>.whl
```

(replace the filename with the actual one produced). On macOS, add `--no-default-features` here as well.

Note that for actually publishing to PyPI you do not build wheels manually; that is handled automatically by the GitHub action described in the next section.

## Deploying to PyPI
Deploying to the [PyPI website](https://pypi.org/project/batss/) is how users are able to install via `pip install batss`. This is done by a GitHub action in .github/workflows/build_and_publish.yml. It builds binary wheels (so that users do not need a Rust compiler to install) for each major platform and Python version and uploads them to PyPI. This is done automatically whenever there is a new GitHub release. The steps are

1. Bump the version number in pyproject.toml, e.g., change 1.0.0 to 1.0.1 if the last uploaded version was 1.0.0 (or bump minor or major version numbers if appropriate according to [semantic versioning](https://semver.org/)). For the rest of this section assume the version number is 1.0.1.

2. Commit this and other changes to the main branch and push to github.

3. On the [Github page](https://github.com/UC-Davis-molecular-computing/batss), click on Releases-->Create a new release. Title the release `v1.0.1` (or whatever is the version number). This can actually be anything but that is a good title. More importantly, press `Tag: Select tag` and type `v1.0.1`; this must be exactly that, i.e., it must be the lowercase letter `v` followed by the version number in pyproject.toml. Press Enter (i.e., don't just click outside once you've typed; you have to press Enter for it to create the tag). Importantly, there must be no tag already called `v1.0.1`; to double-check, click on Tags at the top of the releases page to see existing tags. If that tag exists, delete it. 

4. Click Publish release.

5. Click on Actions at the top of the page to ensure that the build_and_publish.yml action is running. Because it must compile Rust files for many different platforms (and for each many different Python versions), this takes several minutes. If you are testing out something repeatedly in a short time, it is best to delete all or most of the parts of the Github action that deal with building, since that will eat up limited computation time alloted by Github.

6. If the action successfully completes, go to https://pypi.org/project/batss/ to verify that it is the latest version, 1.0.1 in this case.

7. Recall above that the tag must not already be present. If you need to redo this due to a mistake, not only must you delete the release first, you must also click on Tags at the top of the Releases page and delete the tag. If you do not, then it will not successfully upload to PyPI.