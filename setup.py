import sys, os
from setuptools import setup, find_packages

'''
Symmetric Play Setup Script

Notes for later: see package_data arg if additional files need to be supplied.

'''
if sys.version_info.major != 3:
    print('Please use Python3!')

setup(name='symmetric_play',
        packages=[package for package in find_packages()
                    if package.startswith('symmetric_play')],
        install_requires=[
            'gym[atari,classic_control]==0.15.3',
            'stable_baselines[mpi]==2.9.0',
            'matplotlib'],
        extras_require={
            'cpu' : ['tensorflow==1.14.0'],
            'gpu' : ['tensorflow-gpu==1.14.0'],
            },
        description='Framework for Symmetric Play Experiments',
        author='Joey Hejna',
        url='https://github.com/jhejna/symmetric_play',
        lisence='MIT',
        version='0.0.1',
        )
