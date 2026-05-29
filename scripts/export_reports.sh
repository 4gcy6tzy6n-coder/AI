#!/bin/bash

# Report Export Script

set -e

echo "Exporting Post Transformer AI reports..."

# Parse arguments
REPORT_TYPE=${1:-all}
FORMAT=${2:-html}
OUTPUT_DIR=${3:-./reports/export}

# Create output directory
mkdir -p "$OUTPUT_DIR"

case $REPORT_TYPE in
    daily)
        echo "Exporting daily report..."
        python -m src.utils.report_generator --type daily --format "$FORMAT" --output "$OUTPUT_DIR/daily_report.$FORMAT"
        ;;
    milestone)
        echo "Exporting milestone report..."
        python -m src.utils.report_generator --type milestone --format "$FORMAT" --output "$OUTPUT_DIR/milestone_report.$FORMAT"
        ;;
    final)
        echo "Exporting final report..."
        python -m src.utils.report_generator --type final --format "$FORMAT" --output "$OUTPUT_DIR/final_report.$FORMAT"
        ;;
    all)
        echo "Exporting all reports..."
        $0 daily "$FORMAT" "$OUTPUT_DIR"
        $0 milestone "$FORMAT" "$OUTPUT_DIR"
        $0 final "$FORMAT" "$OUTPUT_DIR"
        ;;
    *)
        echo "Unknown report type: $REPORT_TYPE"
        echo "Usage: $0 [daily|milestone|final|all] [html|pdf|md] [output_dir]"
        exit 1
        ;;
esac

echo "Reports exported to $OUTPUT_DIR"
