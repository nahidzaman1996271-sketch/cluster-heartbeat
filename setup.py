"""
Setup script for Cluster Heartbeat.
AI-powered GPU cluster intelligence system.
"""

from setuptools import setup, find_packages
from pathlib import Path
import os

# Read the contents of README file
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "AI-powered GPU cluster intelligence system - One Signal, Three Outcomes"

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "torch>=1.12.0",
        "fastapi>=0.85.0",
        "uvicorn[standard]>=0.18.0",
        "pydantic>=1.10.0",
        "pyyaml>=6.0",
        "python-dotenv>=0.19.0",
        "click>=8.1.0",
        "python-json-logger>=2.0.0",
        "tqdm>=4.64.0",
    ]

# Read version from config or use default
version = "1.0.0"
try:
    config_path = Path(__file__).parent / "config" / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            version = config.get("project", {}).get("version", "1.0.0")
except:
    pass

setup(
    # Basic package information
    name="cluster-heartbeat",
    version=version,
    description="AI-powered GPU cluster intelligence system - One Signal, Three Outcomes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # Author information
    author="Nahid Ibn Zaman, Farhan Ishraq Ifti, Tahmid Rashid Pranjol",
    author_email="your-email@example.com",
    maintainer="Slow Walker Team",
    maintainer_email="your-email@example.com",
    
    # URLs
    url="https://github.com/your-username/cluster-heartbeat",
    download_url="https://github.com/your-username/cluster-heartbeat/archive/main.tar.gz",
    
    # Project metadata
    project_urls={
        "Documentation": "https://github.com/your-username/cluster-heartbeat/wiki",
        "Source Code": "https://github.com/your-username/cluster-heartbeat",
        "Issue Tracker": "https://github.com/your-username/cluster-heartbeat/issues",
        "Hackathon Submission": "https://github.com/your-username/cluster-heartbeat",
    },
    
    # Package configuration
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "docs"]),
    include_package_data=True,
    package_data={
        "cluster_heartbeat": [
            "config/*.yaml",
            "config/*.yml",
            "data/*.csv",
            "data/raw/*",
            "models_checkpoints/*.pt",
            "models_checkpoints/*.pkl",
        ],
    },
    
    # Dependencies
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.19.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
            "mkdocs>=1.4.0",
            "mkdocs-material>=8.5.0",
            "memory-profiler>=0.61.0",
        ],
        "gpu": [
            "torch>=1.12.0 --index-url https://download.pytorch.org/whl/cu118",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.9.0",
        ],
        "monitoring": [
            "prometheus-client>=0.14.0",
            "grafana-api>=1.0.0",
        ],
        "kubernetes": [
            "kubernetes>=25.3.0",
            "kubernetes-asyncio>=24.0.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.19.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
            "jupyter>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.9.0",
            "prometheus-client>=0.14.0",
            "kubernetes>=25.3.0",
        ],
    },
    
    # Entry points - console scripts
    entry_points={
        "console_scripts": [
            "cluster-heartbeat=src.main:run_cli",
            "ch-train=src.training.train:main",
            "ch-api=src.api.main:main",
            "ch-test=src.main:test_cluster_heartbeat",
        ],
        "pytest11": {
            "cluster_heartbeat_tests": "tests.conftest",
        },
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: System Administrators",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Monitoring",
        "Topic :: System :: Distributed Computing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Framework :: FastAPI",
        "Framework :: PyTorch",
        "Natural Language :: English",
    ],
    
    # Keywords for PyPI
    keywords=[
        "gpu-cluster",
        "ai-ops",
        "kubernetes-scheduler",
        "anomaly-detection",
        "cost-optimization",
        "prometheus",
        "dcgm",
        "deep-learning",
        "pytorch",
        "fastapi",
        "hackathon",
    ],
    
    # License
    license="MIT",
    
    # Platforms
    platforms=["any"],
    
    # Additional metadata
    zip_safe=False,
)