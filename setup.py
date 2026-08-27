#!/usr/bin/env python

from setuptools import setup

from pathlib import Path

HERE = Path(__file__).resolve().parent

_ns = {"__file__": str(HERE / "zeus21" / "_version.py")}
exec((HERE / "zeus21" / "_version.py").read_text(), _ns)
VERSION = _ns["get_version"]()


setup(
    name='zeus21',
          version=VERSION,
          description='Zeus21: An analytic 21-cm code for cosmic dawn and EoR.',
          url='https://github.com/JulianBMunoz/Zeus21',
          author='Julian B. Muñoz',
          author_email='julianmunoz@austin.utexas.edu',
          license='MIT',
          packages=['zeus21'],
          long_description=open('README.md').read(),
          install_requires=[
           "numpy>=2.0",
           "scipy",
           "mcfit",
           "classy",
           "numexpr",
           "astropy",
           "powerbox",
           "pyfftw",
           "tqdm"
       ],
)
