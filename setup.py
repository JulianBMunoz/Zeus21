#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='zeus21_mQDM',
    version='0.1dev',
    description='Zeus21: An analytic 21-cm code for cosmic dawn and EoR.',
    url='https://github.com/JulianBMunoz/Zeus21',
    author='Julian B. Muñoz',
    author_email='julianmunoz@austin.utexas.edu',
    license='MIT',
    packages=['zeus21_mQDM'],
    include_package_data=True,
    package_data={'zeus21_mQDM': ['data/*']},
    long_description=open('README.md').read(),
    install_requires=[
        "numpy",
        "scipy",
        "mcfit",
        "classy",
        "numexpr",
        "astropy",
       ],
)
