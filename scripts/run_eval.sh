#!/bin/bash

# Evaluation Runner Script

set -e

echo "Running Post Transformer AI evaluation..."

# Parse arguments
EVAL_TYPE=${1:-prototype}
OUTPUT_DIR=${2:-./reports}

# Create output directory
mkdir -p "$OUTPUT_DIR"

case $EVAL_TYPE in
    prototype)
        echo "Running prototype evaluation..."
        python -m experiments.exp_001_unit_pipeline.evaluate --output "$OUTPUT_DIR/prototype_eval.json"
        ;;
    ablation)
        echo "Running ablation study..."
        python -m experiments.exp_002_memory_gates.ablation --output "$OUTPUT_DIR/ablation_results.json"
        ;;
    calibration)
        echo "Running calibration..."
        python -m experiments.exp_003_tsla_calibration.calibrate --output "$OUTPUT_DIR/calibration_results.json"
        ;;
    complexity)
        echo "Running complexity evaluation..."
        python -m experiments.exp_006_complexity.evaluate --output "$OUTPUT_DIR/complexity_results.json"
        ;;
    all)
        echo "Running all evaluations..."
        $0 prototype "$OUTPUT_DIR"
        $0 ablation "$OUTPUT_DIR"
        $0 calibration "$OUTPUT_DIR"
        $0 complexity "$OUTPUT_DIR"
        ;;
    *)
        echo "Unknown evaluation type: $EVAL_TYPE"
        echo "Usage: $0 [prototype|ablation|calibration|complexity|all] [output_dir]"
        exit 1
        ;;
esac

echo "Evaluation completed! Results saved to $OUTPUT_DIR"
