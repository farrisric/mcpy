from setuptools import setup, find_packages

setup(
    name="mcpy",
    version="0.1.0",
    description="A python package to run atomistic Monte Carlo simulations",
    packages=find_packages(),
    install_requires=[
        "ase>=3.23.0",
        "mace-torch>=0.3.9",
#        "mpi4py>=4.0.1",
    ],
    python_requires='>=3.11',
)
