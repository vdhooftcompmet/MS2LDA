#!/bin/bash

# Ensure the script exits if any command fails
set -e

# Get the directory where this script is located (should be the project root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Set the PYTHONPATH to include the project root directory
# This allows python to find the MS2LDA package when running the script inside App/
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

echo "PYTHONPATH set to: $PYTHONPATH"
echo "Running MS2LDA CLI script..."

# Execute the Python script located in App/, passing all arguments ($@) received by this bash script
python "${SCRIPT_DIR}/scripts/ms2lda_runfull.py" "$@"

echo "CLI script finished."
