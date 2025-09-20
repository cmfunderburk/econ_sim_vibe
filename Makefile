# Economic Simulation Development Makefile
.PHONY: help install install-dev test test-fast lint format check clean validate figures

# Default target
help:
	@echo "Economic Simulation Development Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo ""
	@echo "Simulation:"
	@echo "  run          Run simulation (CONFIG=config/file.yaml SEED=42)"
	@echo ""
	@echo "Testing:"
	@echo "  test         Run full test suite"
	@echo "  test-fast    Run tests with parallel execution"
	@echo "  validate     Run economic validation scenarios"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint         Run linting (flake8, mypy)"
	@echo "  format       Format code (black, ruff)"
	@echo "  check        Run all quality checks"
	@echo ""
	@echo "Analysis:"
	@echo "  figures      Regenerate plots from Parquet logs"
	@echo "  profile      Profile performance bottlenecks"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean        Remove temporary files and caches"

# Installation
install:
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt
	pre-commit install

# Simulation
run:
	@echo "Running simulation with CONFIG=$(CONFIG) SEED=$(SEED)"
	OPENBLAS_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 python scripts/run_simulation.py --config $(CONFIG) --seed $(SEED)

# Testing
test:
	OPENBLAS_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 pytest tests/ -v

test-fast:
	OPENBLAS_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 pytest tests/ -v -n auto

validate:
	OPENBLAS_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 pytest tests/validation/ -v --tb=short

# Code quality
lint:
	flake8 src/ tests/ scripts/
	mypy src/ --ignore-missing-imports
	ruff check src/ tests/ scripts/

format:
	black src/ tests/ scripts/
	ruff format src/ tests/ scripts/

check: lint test-fast

# Analysis and visualization
figures:
	@echo "Regenerating figures from simulation logs..."
	python scripts/generate_figures.py --input simulation_results/ --output figures/

profile:
	@echo "Running performance profiling..."
	python -m cProfile -o profile.stats scripts/run_simulation.py --config config/performance_test.yaml
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Maintenance
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -f profile.stats

# Environment reproducibility
requirements.lock: requirements.txt
	pip-compile --generate-hashes requirements.txt

requirements-dev.lock: requirements-dev.txt requirements.txt
	pip-compile --generate-hashes requirements-dev.txt

# CI/CD helpers
ci-test:
	@echo "Running CI test suite with deterministic settings..."
	OPENBLAS_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 pytest tests/ --tb=short --maxfail=5

ci-validate:
	@echo "Running validation scenarios for CI..."
	OPENBLAS_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 pytest tests/validation/ --tb=line --maxfail=1