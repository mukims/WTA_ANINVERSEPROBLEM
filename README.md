# Generation scripts for Identifying Winner-Takes-All Emergence in Random Nanowire Networks: an Inverse Problem

Repository for the generation of nanowire networks (NWN) and calculation of conductance evolution curves as a function of junction parameters (resistivity, Aj and alpha_j).

## Files

File **"geometry_generator.py"** generates random NWNs with predefined constants based on imaging of real networks. It ensures that the generated NWN will have a defined ammount of junctions.

File **"mnr_cond_adapted.py"** calculates the conductance evolution based on the input geometry. In this file are set all the junction parameters responsible for the evolution. This script has a dependency on the 
minimum residual routine to solve the system of equations, which was taken from:

https://web.stanford.edu/group/SOL/software/minres/
C. C. Paige and M. A. Saunders (1975),
Solution of sparse indefinite systems of linear equations,
SIAM J. Numer. Anal. 12(4), pp. 617-629. 

File **"evolve.py"** provides a higher level of control of the previous files for massive generation of configurations in order to later build configurational averages (CAs) as well as input configurations.

File **"power_law_surf.ipynb"** provides the script that illustrates the single-junction power-law.
