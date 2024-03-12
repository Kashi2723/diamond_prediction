from setuptools import find_packages,setup
from typing import List

setup(
    name='provide name',
    packages=find_packages(where = "src"),
    package_dir= {"":"src"}
)