# ============================================================================
# Makefile for PPO-GNN Project
# ============================================================================

.PHONY: help install install-dev test clean format lint docs

help:
	@echo "Available commands:"
	@echo "  make install       - Install package and dependencies"
	@echo "  make install-dev   - Install with development dependencies"
	@echo "  make test          - Run tests"
	@echo "  make clean         - Remove build artifacts"
	@echo "  make format        - Format code with black"
	@echo "  make lint          - Run linting checks"
	@echo "  make docs          - Build documentation"
	@echo "  make train         - Train PPO-GNN model"
	@echo "  make benchmark     - Run benchmark comparison"

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

format:
	black src/ experiments/ tests/
	isort src/ experiments/ tests/

lint:
	flake8 src/ experiments/
	mypy src/
	pylint src/

docs:
	cd docs && make html

train:
	python src/training/train_ppo_gnn.py --config configs/ppo_gnn_config.yaml

benchmark:
	python experiments/benchmark_comparison.py --output results/benchmark/
