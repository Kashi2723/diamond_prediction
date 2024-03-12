from setuptools import find_packages,setup
from typing import List

setup(
    name='Diamond_Prediction',
    packages=find_packages(where = "src"),
    package_dir= {"":"src"}
)