"""Classic setup.py kept for compatibility with older tooling.

The authoritative project metadata lives in ``pyproject.toml``; this
file exists only so environments that still rely on ``python setup.py
install``-style workflows can install the package. New configuration
should be added to ``pyproject.toml`` rather than here.
"""

from setuptools import find_packages, setup

setup(
    name="taxi-demand-forecast",
    version="0.2.0",
    description=(
        "Hourly NYC HVFHV pickup demand forecaster: data loading, "
        "feature engineering, scikit-learn models, and evaluation "
        "utilities."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Emily Wang, Yuhe Jiang, Runhua Cao",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(include=["taxi_demand", "taxi_demand.*"]),
    include_package_data=True,
    install_requires=[
        "pandas>=1.5",
        "numpy>=1.23",
        "scikit-learn>=1.1",
        "pyarrow>=10.0",
        "requests>=2.28",
        "matplotlib>=3.5",
        "joblib>=1.2",
        "beautifulsoup4>=4.11",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "coverage>=7.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
