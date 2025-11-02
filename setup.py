# ============================================================================
# Setup Script for PPO-GNN Package
# ============================================================================

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() 
        for line in fh 
        if line.strip() and not line.startswith("#") and not line.startswith("--")
    ]

setup(
    name="ppo-gnn-fuel-delivery",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@university.edu",
    description="Constraint-Aware PPO-GNN for Stochastic Fuel Delivery Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YourUsername/PPO-GNN-Fuel-Delivery",
    project_urls={
        "Bug Tracker": "https://github.com/YourUsername/PPO-GNN-Fuel-Delivery/issues",
        "Documentation": "https://github.com/YourUsername/PPO-GNN-Fuel-Delivery/docs",
        "Source Code": "https://github.com/YourUsername/PPO-GNN-Fuel-Delivery",
        "Paper": "https://arxiv.org/abs/XXXX.XXXXX",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.12b0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ],
        "gurobi": [
            "gurobipy>=10.0.0",
        ],
        "wandb": [
            "wandb>=0.12.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "ipywidgets>=7.6.0",
        ],
        "docs": [
            "sphinx>=4.3.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ppo-gnn-train=src.training.train_ppo_gnn:main",
            "ppo-gnn-eval=src.evaluation.evaluate:main",
            "ppo-gnn-benchmark=experiments.benchmark_comparison:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
