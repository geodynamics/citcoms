# CitcomS with Data Assimilation

## Installation

This version of CitcomS only requires a valid MPI distribution (no more python / pyre).  Most clusters can load an MPI distribute using `modules`

For example, at the University of Bern I use:

```module load iomkl/2018b```

For reasons relating to compilers, the cluster and node setup, some mpi versions may be preferred for your particular cluster.

To install CitcomS v3.3.1 with data assimilation go into the src/ directory and execute

```./mymake.py```

[Yes OK, for the step above you do need a python distribution, but this is only to run the commands that configure and make Citcoms.  Python is not actually used for running the code]

Example input and output configuration files are provided in:

```examples/```

This example is a good one to try and run first:

```src/examples/Full```

You will need to setup your job submission script.  See a slurm example in jobsubmit/  Please add your own submission scripts to the same directory so we have more examples

## Features that are implemented

A data assimilation manual is hosted in an svn repository located at ```https://svn.gps.caltech.edu/repos/docs/data_assimilation```.  This manual contains information about the parameters that are used, both for the convection code and also the pre- and post-processing scripts.  Access to the svn can be requested.  Some options in the manual might be out-dated, so prioritise the options in this README.

### Slab and lithosphere assimilation (grep for 'DJB SLAB')
1. ```lith_age_depth_function``` (bool)
1. ```lith_age_exponent``` (double)
1. ```lith_age_min``` (double)
1. ```lith_age_stencil_value``` (double)
1. ```slab_assim``` (bool)
1. ```slab_assim_file``` (char)
1. ```sten_temp``` as an output option (char)

### Composition (grep for 'DJB COMP')
1. ```hybrid_method``` (bool)
1. increased memory for tracer arrays (icushion parameter)
1. turn off tracer warnings using ```itracer_warnings=off``` in input cfg file

### Viscosity structures (grep for 'DJB VISC')
1. case 20, used in Flament et al. (2013, 2014)
1. case 21, used in Flament et al. (2014), model TC8
1. case 22, used in Flament et al. (2014), model TC7
1. case 23, used in Flament et al. (2014), model TC9
1. case 24, used in Zhang et al. (2010) and Bower et al. (2013)
1. case 25, used by Flament for Extendend-Boussinesq (EBA) models
1. case 26, used by Flament for EBA models
1. case 27, used by Flament for EBA models
1. case 28, used by Flament for EBA models
1. case 112, used by Hassan
1. case 113, used by Hassan
1. case 117, used by Hassan
1. case 118, used by Hassan

### Output time (grep for 'DJB TIME')
1. output data in regular increments of age (Myr) as well as/rather than number of time steps
    - ```storage_spacing_Myr``` (int)
    - if you only want to output data by age (Ma), you should set ```storage_spacing``` to a large integer value in order to suppress the regular time outputs
    - both ```storage_spacing_Myr``` and ```storage_spacing``` can be used together, in which case data is output whenever either one of these output criteria is satisfied
1. exit time loop when the model reaches negative ages (currently hard-coded to be <0 Ma)
    - ```exit_at_present``` (bool)
    
### Extended-Boussinesq modifications (grep for 'DJB EBA')
1. depth-dependent scaling for the dissipation number (Di)
    - Di is typically defined at the surface, using surface values of alpha, cp, and g
    - a depth-dependent scaling (typically between 0 and 1) is introduced to scale Di
    - this scales Di in the energy equation, i.e. the adiabatic, viscous, and latent heating terms
    - useful to avoid intense shear heating near the surface and hence avoid artefacts when using data assimilation  
2. Therefore, an extra column is added to the ```refstate_file```:
    - [column 1] rho
    - [column 2] gravity
    - [column 3] thermal expansion coefficient (alpha)
    - [column 4] heat capacity
    - [column 5] dissipation number scaling

### Output (grep for 'DJB OUT')
1. Composition and temperature spherical harmonics as an output option (char).  See issue, since output for comp only occurs for the last comp field.
    - ```comp_sph```
    - ```temp_sph```
    - ```sten_temp```
1. Fixed pid file output for lith_age_min, z_interface, and output_optional

### Topography (grep for 'DJB TOPO')
1. Parameter to remove buoyancy above a given znode for computing dynamic topography
    - ```remove_buoyancy_above_znode``` (int)

### Ultra-low velocity zone (grep for 'DJB ULVZ")
1. Modifications to enable ULVZ modelling as in Bower et al. (2011).  These amendments will not affect the data assimilation.
    - domain volumes for absolute tracer method for different initial ULVZ volumes
    - permeable domain for regional models
    - sine perturbation for initial temperature

## Dan's work area follows

### Rakib's code

Sent to me by Sabin (12/09/17) CitcomSModDtopoRelease\_Rakib.zip
See diff/ directory for complete record

1. Advection\_diffusion
    - Scaled visc and adiabatic heating by Di (COMPLETE)
1. convection\_variables.h
    - blob\_profile
    - silo parameters
    - mantle\_temp\_adiabatic\_increase
1. global\_defs.h
    - Shell-output facility
1. Initial\_temperature.c
    - bunch of silo / blob related functions  
    - evidentally seeding plumes for the IC
1. Material\_properties.c
    - refstate updates (COMPLETE)
1. Output.c
    - outputs of shells (theta, phi, r, temperature, vr)
1. Output\_vtk.c
    - perhaps not related to Rakib's work
1. Viscosity\_structures.c
    - some new viscosity structures (case 112, 113, 117, 118) (COMPLETE)

### Code features NOT implemented

1. internal velocity bcs (I don't think these are used by anyone anyway)
1. outputs of heating terms, divv
1. tracer density for elements and nodes output (added by Ting, see svn r52 through r55)
1. reverse gravity acceleraton (added by Ting, see svn r76) for SBI
   note that this appears to have been subsequently removed in r88?

### Log of features implemented from legacy code
1. Adv\_diff -> COMPLETE
1. BC\_util -> ivels (TODO)
1. Composition\_related.c -> tracer density (TODO)
1. composition\_related.h -> tracer density (TODO)
1. Convection.c -> COMPLETE
1. Element\_calculations -> COMPLETE
1. Full\_boundary\_conditions -> ivels (TODO)
1. Full\_lith\_age\_read\_files -> COMPLETE
1. Full_read_input_from_file -> ivels (TODO)
1. Full\_solver.c -> COMPLETE
1. global\_defs.h -> ivels (TODO)
1. Instructions.c -> outputs, heating terms, divv (TODO)
1. Lith\_age.c -> COMPLETE
1. output.c -> various outputs (TODO)
1. Pan\_problem.c -> COMPLETE
1. Problem\_related.c -> ivels (TODO)
1. Regional\_bcs.c -> ivels (TODO)
1. Regional\_lith\_age\_read\_files -> COMPLETE
1. Regional\_read\_input\_files -> ivels (TODO)
1. Regional\_solver -> COMPLETE
1. solver.h -> COMPLETE
1. Stokes\_flow\_incom -> divv calculation for output (TODO)
1. tracer\_defs.h -> output tracer density (TODO)
1. Viscosity\_structure.c -> COMPLETE
