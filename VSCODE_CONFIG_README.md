# VSCode Configuration for ppsim-rust

This document explains the VSCode configuration that has been set up to resolve the issue with Conda and virtual environment conflicts when using maturin.

## Problem

When both a Python virtual environment (VIRTUAL_ENV) and a Conda environment (CONDA_PREFIX) are active simultaneously, maturin fails with the error:

```
💥 maturin failed
  Caused by: Both VIRTUAL_ENV and CONDA_PREFIX are set. Please unset one of them
```

## Solution

A comprehensive solution has been implemented that automatically handles the environment setup when a terminal is opened in VSCode:

### 1. Custom PowerShell Profile

Created a custom PowerShell profile at `.vscode/powershell/Microsoft.VSCode_profile.ps1` that:

```powershell
# Custom PowerShell profile for VSCode terminals
# Unset CONDA_PREFIX to prevent conflicts with virtual environments
if ($env:CONDA_PREFIX) {
    Remove-Item Env:\CONDA_PREFIX -ErrorAction SilentlyContinue
}

# Prevent Conda from auto-activating
$env:CONDA_AUTO_ACTIVATE_BASE = "false"

# Activate the virtual environment if it exists
$venvPath = Join-Path $PSScriptRoot -ChildPath "../../.venv"
if (Test-Path $venvPath) {
    $activateScript = Join-Path $venvPath -ChildPath "Scripts/Activate.ps1"
    if (Test-Path $activateScript) {
        . $activateScript
    }
}
```

This profile:
- Automatically unsets CONDA_PREFIX when a terminal opens
- Prevents Conda from auto-activating its base environment
- Activates the project's virtual environment

### 2. VSCode Terminal Configuration

Updated `.vscode/settings.json` to use the custom PowerShell profile:

```json
{
    "python.analysis.extraPaths": [
        "./python"
    ],
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe",
    "python.terminal.activateEnvironment": false,
    "terminal.integrated.env.windows": {
        "CONDA_AUTO_ACTIVATE_BASE": "false"
    },
    "terminal.integrated.defaultProfile.windows": "PowerShell",
    "terminal.integrated.profiles.windows": {
        "PowerShell": {
            "source": "PowerShell",
            "icon": "terminal-powershell",
            "args": [
                "-NoExit",
                "-ExecutionPolicy", "Bypass",
                "-NoLogo",
                "-NoProfile",
                "-File", "${workspaceFolder}/.vscode/powershell/Microsoft.VSCode_profile.ps1"
            ]
        }
    }
}
```

These settings:
- Configure VSCode to use our custom PowerShell profile for all terminals
- Set the Python interpreter to use the virtual environment
- Disable VSCode's built-in environment activation (we handle it in our profile)
- Prevent Conda from auto-activating

## Using the Configuration

1. **Restart VSCode** for the settings to take effect
2. Open a new terminal in VSCode - it will:
   - Show the (.venv) prefix in the prompt
   - Have CONDA_PREFIX unset
   - Have the virtual environment activated
3. Run maturin directly:

   ```powershell
   maturin develop -r --features flm
   ```

## Troubleshooting

If the issue persists after restarting VSCode:

1. Verify that the terminal shows the (.venv) prefix in the prompt
2. Try closing all VSCode windows and reopening the project
3. Check if there are any other PowerShell profiles that might be interfering
