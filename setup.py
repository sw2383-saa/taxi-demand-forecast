from setuptools import setup, find_packages

setup(
    name="taxi-demand-forecast",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "pyarrow",
        "requests",
        "matplotlib",
        "joblib",
    ],
)
