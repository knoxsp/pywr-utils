#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='pywr utils',
    version='0.1',
    description='Utilities for manipulating pywr models',
    packages=find_packages(),
    package_data={},
    include_package_data=True,
    entry_points='''
    [console_scripts]
    pywr-utils=pywr_utils.cli:start_cli
    ''',
)
