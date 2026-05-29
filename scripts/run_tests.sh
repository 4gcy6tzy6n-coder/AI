#!/bin/bash

# Test Runner Script

set -e

echo "Running Post Transformer AI tests..."

# Parse arguments
TEST_TYPE=${1:-all}
COVERAGE=${2:-false}

# Run based on test type
case $TEST_TYPE in
    unit)
        echo "Running unit tests..."
        if [ "$COVERAGE" = "true" ]; then
            pytest tests/unit/ -v --cov=src --cov-report=term-missing
        else
            pytest tests/unit/ -v
        fi
        ;;
    contract)
        echo "Running contract tests..."
        pytest tests/contract/ -v
        ;;
    integration)
        echo "Running integration tests..."
        pytest tests/integration/ -v
        ;;
    e2e)
        echo "Running end-to-end tests..."
        pytest tests/e2e/ -v
        ;;
    regression)
        echo "Running regression tests..."
        pytest tests/regression/ -v
        ;;
    safety)
        echo "Running safety tests..."
        pytest tests/safety/ -v
        ;;
    performance)
        echo "Running performance tests..."
        pytest tests/performance/ -v
        ;;
    all)
        echo "Running all tests..."
        if [ "$COVERAGE" = "true" ]; then
            pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
        else
            pytest tests/ -v
        fi
        ;;
    *)
        echo "Unknown test type: $TEST_TYPE"
        echo "Usage: $0 [unit|contract|integration|e2e|regression|safety|performance|all] [true|false]"
        exit 1
        ;;
esac

echo "Tests completed!"
