# When the powershell terminal opens, this profile activates the virtual environment 
# and sets pip to be quiet.

# Activate the virtual environment if it exists
$venvPath = Join-Path (Split-Path -Parent $PSScriptRoot) -ChildPath "../.venv"
"Checking for virtual environment at: $venvPath"

if (Test-Path $venvPath) {
    $activateScript = Join-Path $venvPath -ChildPath "Scripts/Activate.ps1"
    "Activate script path: $activateScript"
    
    if (Test-Path $activateScript) {
        # Activate the virtual environment
        . $activateScript
        "Virtual environment activated successfully"
    } else {
        "Activation script not found"
    }
} else {
    "Virtual environment not found"
}

# This avoids all the "Requirement already satisfied" messages when 
# installing the ppsim package locally when running `maturin develop`.
$env:PIP_QUIET = 1
