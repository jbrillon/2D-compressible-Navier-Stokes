**Date written: December 2018**

As part of final project for ME 614 (Computational Fluid Dynamics) at Purdue University taught by Professor Scalo of the Mechanical Engineering department. 

# 2D Compressible Navier-Stokes Solver

Solves the 2D compressible Navier-Stokes equations using finite-difference with a staggered grid arrangement to simulate acoustic waves in a doubly periodic domain.

## Evolution of Pressure in Acoustic Waves

The pressure fluctuations of acoustic waves in a doubly periodic domain are shown below.

<img src="https://raw.githubusercontent.com/jbrillon/2D-compressible-Navier-Stokes/master/Figures/PressureContour/DoublyPeriodic/pressure_CFL075_with_HT_and_viscous.gif" width="45%"></img>
<img src="https://raw.githubusercontent.com/jbrillon/2D-compressible-Navier-Stokes/master/Figures/PressureSlice/DoublyPeriodic/pressureSlice_CFL075_with_HT_and_viscous.gif" width="45%"></img>

## Order of Accuracy Verification

Below illustrates the expected 2nd-order convergence of the staggered finite-difference scheme is acheived. 

<img src="https://raw.githubusercontent.com/jbrillon/2D-compressible-Navier-Stokes/master/Figures/OoA/DoublyPeriodic/RMS_error_euler_x_ymomentum.png" width="45%"></img>
<img src="https://raw.githubusercontent.com/jbrillon/2D-compressible-Navier-Stokes/master/Figures/OoA/DoublyPeriodic/RMS_error_euler_y_xmomentum.png" width="45%"></img>

## Staggered Grid Arrangement

The grid arrangement for x-velocity, y-velocity, and passive scalars are shown below, respectfully. 

<img src="https://raw.githubusercontent.com/jbrillon/2D-compressible-Navier-Stokes/master/Figures/GRID/grid_x-velocity.png" width="30%"></img>
<img src="https://raw.githubusercontent.com/jbrillon/2D-compressible-Navier-Stokes/master/Figures/GRID/grid_y-velocity.png" width="30%"></img>
<img src="https://raw.githubusercontent.com/jbrillon/2D-compressible-Navier-Stokes/master/Figures/GRID/grid_pressure.png" width="30%"></img>

## Finite-Difference Operators

The sparsity pattern for the finite-difference operators for the x-momentum, y-momentum, and energy equations are shown below, respectfully.

<img src="https://raw.githubusercontent.com/jbrillon/2D-compressible-Navier-Stokes/master/Figures/SPY/DoublyPeriodic/SPY_matrix_ddx_Periodic_xmomentum.png" width="30%"></img>
<img src="https://raw.githubusercontent.com/jbrillon/2D-compressible-Navier-Stokes/master/Figures/SPY/DoublyPeriodic/SPY_matrix_ddx_Periodic_ymomentum.png" width="30%"></img>
<img src="https://raw.githubusercontent.com/jbrillon/2D-compressible-Navier-Stokes/master/Figures/SPY/DoublyPeriodic/SPY_matrix_ddx_Periodic_energy.png" width="30%"></img>
<img src="https://raw.githubusercontent.com/jbrillon/2D-compressible-Navier-Stokes/master/Figures/SPY/DoublyPeriodic/SPY_matrix_ddy_Periodic_xmomentum.png" width="30%"></img>
<img src="https://raw.githubusercontent.com/jbrillon/2D-compressible-Navier-Stokes/master/Figures/SPY/DoublyPeriodic/SPY_matrix_ddy_Periodic_ymomentum.png" width="30%"></img>
<img src="https://raw.githubusercontent.com/jbrillon/2D-compressible-Navier-Stokes/master/Figures/SPY/DoublyPeriodic/SPY_matrix_ddy_Periodic_energy.png" width="30%"></img>