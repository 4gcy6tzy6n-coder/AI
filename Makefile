.PHONY: help install test lint format clean run-exp

PYTHON := python
PIP := pip

help:
	@echo "Available targets:"
	@echo "  install     - Install dependencies"
	@echo "  test        - Run all tests"
	@echo "  test-unit   - Run unit tests"
	@echo "  test-int    - Run integration tests"
	@echo "  lint        - Run linters"
	@echo "  format      - Format code"
	@echo "  clean       - Clean build artifacts"
	@echo "  run-exp     - Run experiment (make run-exp EXP=exp_001_unit_pipeline)"

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-int:
	pytest tests/integration/ -v

test-contract:
	pytest tests/contract/ -v

lint:
	ruff check src tests
	mypy src

format:
	black src tests
	isort src tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run-exp:
	@if [ -z "$(EXP)" ]; then \
		echo "Error: EXP variable not set. Usage: make run-exp EXP=exp_001_unit_pipeline"; \
		exit 1; \
	fi
	$(PYTHON) -m experiments.$(EXP).run

calibrate:
	$(PYTHON) scripts/run_calibration.sh

eval:
	$(PYTHON) scripts/run_eval.sh

bootstrap:
	bash scripts/bootstrap_repo.sh
