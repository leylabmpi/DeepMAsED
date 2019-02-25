#!/usr/bin/env python
from setuptools import setup, find_packages
import os
import glob
import numpy

    
# dependencies
install_reqs = [
    'numpy>=1.10',
    'pandas>=0.18'
]

## install main application
desc = 'Deep learning for Metagenome Assembly Error Detection (DeepMAsED)'
setup(
    name = 'DeepMAsED',
    version = '0.1.0',
    description = desc,
    long_description = desc + '\n See README for more information.',
    author = 'Nick Youngblut',
    author_email = 'nyoungb2@gmail.com',
    entry_points={
        'console_scripts': [
            'DeepMAsED = DeepMAsED.__main__:main'
        ]
    },
    install_requires = install_reqs,
    include_dirs = [numpy.get_include()],
    license = "MIT license",
    packages = find_packages(),
    package_dir={'DeepMAsED':
                 'DeepMAsED'},
    url = 'https://github.com/nick-youngblut/DeepMAsED'
)




