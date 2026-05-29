#!/bin/bash

# Calibration Runner Script

set -e

echo "Running Post Transformer AI calibration..."

# Parse arguments
CALIBRATION_TYPE=${1:-tsla}
ITERATIONS=${2:-100}

case $CALIBRATION_TYPE in
    tsla)
        echo "Running TSLA calibration..."
        python -m src.core.tsla.calibrate --iterations "$ITERATIONS"
        ;;
    gates)
        echo "Running gate calibration..."
        python -m src.core.gates.calibrate --iterations "$ITERATIONS"
        ;;
    memory)
        echo "Running memory calibration..."
        python -m src.core.memory.calibrate --iterations "$ITERATIONS"
        ;;
    all)
        echo "Running all calibrations..."
        $0 tsla "$ITERATIONS"
        $0 gates "$ITERATIONS"
        $0 memory "$ITERATIONS"
        ;;
    *)
        echo "Unknown calibration type: $CALIBRATION_TYPE"
        echo "Usage: $0 [tsla|gates|memory|all] [iterations]"
        exit 1
        ;;
esac

echo "Calibration completed!"
