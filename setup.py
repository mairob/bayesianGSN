import os

from setuptools import find_packages, setup

# Optional: read the long description from README.md
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="bayesianGSN",
    version="0.0.1",
    description=(
        "Proof-of-concept implementation of a Bayesian Network-based probabilistic GSN tree."
    ),
    author="Robert Maier",
    author_email="maier.rob92@gmail.com",
    license="MIT",
    packages=find_packages(),
    package_data={
        "bayesiangsn.data": ["*.yaml"],
    },
    include_package_data=True,
    install_requires=[
        "matplotlib>=3.9",
        "networkx>=3.2",
        "numpy==1.26.4",
        "pandas>=2.2",
        "pgmpy>=0.1.24",
        "pytest>=8.2",
        "setuptools>=71.1",
        "Sphinx>=7.4",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
)
