#!/usr/bin/env python
from setuptools import setup, find_packages
import os
import glob
import numpy


# dependencies
install_reqs = [
    'numpy>=1.17.0',
    'scipy>=1.3.1',
    'tensorflow>=2.0',
    'tensorboard',
    'keras',
    'scikit-learn',
    'ipython'
]

# getting version from __main__.py
__version__ = None
with open(os.path.join('DeepMAsED', '__main__.py')) as inF:
    for x in inF:
        if x.startswith('def main'):
            break
        if x.startswith('__version__'):
            __version__ = x.split(' ')[2].rstrip().strip("'")
            
## install main application
desc = 'Deep learning for Metagenome Assembly Error Detection'
setup(
    name = 'DeepMAsED',
    version = __version__,
    description = desc,
    long_description = desc + '\n See README for more information.',
    author = 'Nick Youngblut',
    author_email = 'nyoungb2@gmail.com',
    package_data={'DeepMAsED': ['Model/deepmased_model.h5',
                                'Model/deepmased_mean_std.pkl']},
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
    url = 'https://github.com/leylabmpi/DeepMAsED'
)




