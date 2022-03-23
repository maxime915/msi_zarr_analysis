# -*- coding: utf-8 -*-

"setup.py adapted from : https://github.com/kennethreitz/setup.py"

from setuptools import setup, find_packages

import msi_zarr_analysis

with open('README.md', encoding='utf-8') as f:
    readme_data = f.read()

with open('LICENSE', encoding='utf-8') as f:
    license_data = f.read()

setup(
    name=msi_zarr_analysis.__name__,
    version=msi_zarr_analysis.VERSION,
    description='Manipulation toolbox for Mass Spectrometry Image encoded in OME-Zarr',
    long_description=readme_data,
    author='Maxime Amodei',
    author_email='Maxime.Amodei@student.uliege.be',
    url=f'https://github.com/maxime915/{msi_zarr_analysis.__name__}',
    license=license_data,
    packages=find_packages(exclude=('tests')),
    entry_points = {
        'console_scripts': [
            f'{msi_zarr_analysis.__name__} = {msi_zarr_analysis.__name__}.cli.entry:main',
            f'{msi_zarr_analysis.__name__}_cli = {msi_zarr_analysis.__name__}.cli.entry:main',
        ],
    }
)
