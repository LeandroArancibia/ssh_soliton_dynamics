# SSH Soliton Dynamics in Polyacetylene

This code was used to obtain the results published in:  
https://journals.aps.org/prb/abstract/10.1103/PhysRevB.106.245426

## Description

This program simulates a trans-polyacetylene chain within the framework of the Su-Schrieffer-Heeger (SSH) Hamiltonian. The system hosts a soliton and is coupled to an external electric field, which drives the soliton motion along the chain under periodic boundary conditions.

To determine the equilibrium position and electronic configuration, a self-consistent minimization method based on the Hellmann-Feynman theorem is employed, as described in the reference above.

For the time evolution:
- Electronic wavefunctions are propagated using the Crank-Nicolson algorithm.
- Atomic positions evolve according to the Verlet integration scheme.

## Scientific Background

The code is mainly based on:

Ono, Y., & Terai, A. (1990). *Motion of charged soliton in polyacetylene due to electric field*. Journal of the Physical Society of Japan, 59(8), 2893–2904.

The main original modification introduced here is the implementation of the Crank-Nicolson algorithm for the electronic dynamics.

## Features

The code allows flexibility to modify:
- Number of sites
- Time step (Δt)
- Total simulation time
- Electronic occupation
- Hopping parameter
- Other physical parameters

## Requirements

- Python 3.12.3
- NumPy
- SciPy
- Matplotlib

## Usage

```bash
python poliacetileno_verlet_crank_nicholson.py
```
## Notes

This code is intended for research and educational purposes in condensed matter physics and computational modeling of conducting polymers.
