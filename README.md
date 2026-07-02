<p align="center">
<img src="docs/images/Zeus21Logo-Horizontal.jpg" width=75% height=75%>
</p>

# Zeus21: Lightning-fast simulations of cosmic dawn and reionization

[![Tests](https://github.com/JulianBMunoz/Zeus21/actions/workflows/python-tests.yml/badge.svg)](https://github.com/JulianBMunoz/Zeus21/actions/workflows/python-tests.yml)
[![codecov](https://codecov.io/gh/JulianBMunoz/Zeus21/branch/main/graph/badge.svg)](https://codecov.io/gh/JulianBMunoz/Zeus21)

Zeus21 encodes the effective model for the 21-cm power spectrum and global signal from [Muñoz 2023a](https://arxiv.org/abs/2302.08506). The goal is to capture all the nonlocal and nonlinear physics of cosmic dawn in a light and fully Pythonic code. Zeus21 takes advantage of the approximate log-normality of the star-formation rate density (SFRD) during cosmic dawn to compute the 21-cm power spectrum analytically. It agrees with more expensive semi-numerical simulations to roughly 10% precision, but has comparably negligible computational cost (~ s) and memory requirements. Now Zeus21 can also predict galaxy UV luminosity functions (UVLFs) and their linear clustering (galaxy bias) at any z, see [Muñoz et al. 2023b](https://arxiv.org/abs/2306.09403) for the implementation and application to JWST. Now Zeus21 includes Population III stars with inhomogeneous Lyman-Werner feedback and relative velocities (with their fluctuations), as described in [Cruz et al. 2024](https://arxiv.org/abs/2407.18294).

Zeus21 (Zippy Early-Universe Solver for 21-cm) pairs well with data from [HERA](https://reionization.org/), but can be used for any 21-cm inference or prediction. Current capabilities include finding the 21-cm power spectrum (at a broad range of k and z), the global signal, IGM temperatures (Tk, Ts, Tcolor), neutral fraction xHI, Lyman-alpha fluxes, and the evolution of the SFRD; all across cosmic dawn z=5-35. Zeus21 can use three different astrophysical models, one of which emulates 21cmFAST, and can vary the cosmology through CLASS. 

Full **documentation** (with [installation instructions](https://zeus21.readthedocs.io/en/latest/installation.html), [tutorials](https://zeus21.readthedocs.io/en/latest/tutorials.html), and [full api description](https://zeus21.readthedocs.io/en/latest/api/modules.html)) can be found on [ReadTheDocs](https://zeus21.readthedocs.io/en/latest/).

