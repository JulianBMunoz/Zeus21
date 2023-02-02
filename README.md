# Zeus21: An analytic code for 21-cm at cosmic dawn

Zeus21 encodes the effective model for the 21-cm power spectrum and global signal from [Muñoz 2023](https://arXiv:XXXX). The goal is to encode all the nonlocal and nonlinear physics of cosmic dawn in a light and fully Pythonic code. Zeus21 takes advantage of the approximate log-normality of the star-formation rate density (SFRD) during cosmic dawn to compute the 21-cm power spectrum fully analytically. It agrees with more expensive semi-numerical simulations to <10% precision, but has much smaller computational cost (~ s) and memory requirements.

Zeus (Zippy Early-Universe Solver) pairs well with data from [HERA](https://reionization.org/), but can be used for any 21-cm inference or prediction. Current capabilities include finding the 21-cm power spectrum (at a broad range of k and z), global signal, IGM temperatures (Tk, Ts, Tcolor), neutral fraction xHI, Lyman-alpha fluxes, and SFRD; all across cosmic dawn z=5-35. Zeus has three different astrophysical models included, one of which emulates 21cmFAST.

For a tutorial see `docs/`. Full documentation in XXXX. 

## Installation

Pip etc

## Citation

If you find this code useful please cite:
XXXXX
and include a link to [this Github](https://github.com/JulianBMunoz/Zeus21).
