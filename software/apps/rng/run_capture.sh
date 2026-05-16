#!/bin/bash
# Ensure the script stops on errors
set -e

echo "Preparing to run RNG automated capture and plotting script..."
echo "--------------------------------------------------------"

# 1. Get the directory of this script (rng folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 2. Automatically find and activate the virtual environment (.venv)
VENV_ACTIVATE_PATH="../../../../.venv/bin/activate"

if [ -f "$VENV_ACTIVATE_PATH" ]; then
    echo "Found virtual environment, activating (.venv)..."
    source "$VENV_ACTIVATE_PATH"
else
    echo "Error: Cannot find virtual environment at $VENV_ACTIVATE_PATH"
    echo "Please ensure you have created the .venv in the Lab directory!"
    exit 1
fi

echo "Installing required packages if needed (pyserial and matplotlib)..."
pip install -q pyserial matplotlib

# 3. Call the Python script
echo "--------------------------------------------------------"
echo "Starting Python script..."
python3 capture_and_plot.py

# 4. Deactivate the virtual environment upon completion
deactivate
echo "--------------------------------------------------------"
echo "Script execution completed. You can now view the generated histogram.png!"
