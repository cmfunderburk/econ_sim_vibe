#!/usr/bin/env python3
"""
Economic Simulation Vibe - Setup Configuration

This package implements agent-based modeling of economic exchange with spatial
frictions in market economies. The simulation models rational agents trading
goods on a configurable spatial grid with centralized marketplace access and
movement costs.
"""

from setuptools import setup, find_packages
import os

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read development requirements from requirements-dev.txt
with open('requirements-dev.txt') as f:
    dev_requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="econ-sim-vibe",
    version="0.1.0",
    author="Economic Simulation Research Team",
    author_email="cmfunderburk@example.com",
    description="Research-grade economic simulation with spatial frictions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cmfunderburk/econ_sim_vibe",
    
    # Package configuration
    packages=find_packages(),
    package_dir={'': '.'},
    
    # Include additional files
    include_package_data=True,
    package_data={
        'config': ['*.yaml'],
        'tests': ['*.py'],
    },
    
    # Dependencies
    install_requires=requirements,
    extras_require={
        'dev': dev_requirements,
        'test': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
        ],
        'visualization': [
            'pygame>=2.0.0',
            'matplotlib>=3.5.0',
        ],
    },
    
    # Entry points for command-line scripts
    entry_points={
        'console_scripts': [
            'econ-sim=scripts.run_simulation:main',
            'econ-validate=scripts.validate_scenario:main',
        ],
    },
    
    # Python version requirement
    python_requires='>=3.8',
    
    # Classifiers for PyPI
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Economics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    
    # Keywords for discoverability
    keywords="economics simulation agent-based spatial walrasian equilibrium",
    
    # Project URLs
    project_urls={
        "Documentation": "https://github.com/cmfunderburk/econ_sim_vibe/blob/main/SPECIFICATION.md",
        "Bug Reports": "https://github.com/cmfunderburk/econ_sim_vibe/issues",
        "Source": "https://github.com/cmfunderburk/econ_sim_vibe",
    },
)