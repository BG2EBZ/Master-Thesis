from setuptools import setup, find_packages

setup(
    name="museum_env",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "museum_env": ["assets/*.xml"],
    },
    install_requires=[
        "gymnasium",
        "mujoco",
        "numpy",
        "scikit-fuzzy",
    ],
)
