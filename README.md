- [CitcomS with Data Assimilation](#citcoms-with-data-assimilation)
    - [Citation](#citation)
    - [Obtaining the code](#obtaining-the-code)
    - [Installation](#installation)
        - [Quick start for a cluster](#quick-start-for-a-cluster)
        - [Quick start for Mac OSX](#quick-start-for-mac-osx)
    - [Examples and User Guide](#examples-and-user-guide)
    - [Code Modifications and Parameter Names](#code-modifications-and-parameter-names)
        - [Slab and lithosphere assimilation](#slab-and-lithosphere-assimilation)
        - [Composition](#composition)
        - [Viscosity structures](#viscosity-structures)
        - [Time output](#time-output)
        - [Extended-Boussinesq modifications](#extended-boussinesq-modifications)
        - [Output fields](#output-fields)
        - [Topography](#topography)
    - [Affiliated codes](#affiliated-codes)
    - [Frequently asked questions (FAQ)](#faq)

# CitcomS with Data Assimilation

This repository *only* contains the C code that implements data assimilation in CitcomS using pregenerated input files.  There are other python scripts that generate the *input* data and these are stored in a different svn repository hosted at Caltech.

## Citation

Bower, D.J., M. Gurnis, and N. Flament (2015), Assimilating lithosphere and slab history in 4-D Earth models, Phys. Earth Planet. Inter., 238, 8-22, doi: 10.1016/j.pepi.2014.10.013.

Open access version: https://eartharxiv.org/9aey5/

## Obtaining the code

Please follow the standard development practice of forking this repository and then cloning your fork.  See, for example:

https://blog.scottlowe.org/2015/01/27/using-fork-branch-git-workflow/

https://help.github.com/articles/fork-a-repo/


## Installation

### Directory structure

- ```src/``` is the CitcomS source code with data assimilation
- ```docs/``` contains a rudimentary (and somewhat outdated) user guide
- ```deprecated/``` contains previous versions of the code that should not be used anymore (but kept for reference)
- ```jobsubmit/``` contains example job submission scripts for cluster environments


### Quick start for a cluster

The following instructions clone this repository, although in practice you'll probably prefer to clone your own fork of this repository.  The instructions also assume you can load an MPI distribution using ```modules```, which is standard on most clusters.  For reasons relating to compilers, the cluster and node setup, some MPI versions may be preferred for your particular cluster.  You should ask your HPC system administrators.  Note that ```module load hdf5``` is only reqired for HDF5 support.  For example, for NCI Australia (Gadi):

```
git clone https://github.com/EarthByte/citcoms.git citcoms_assim
cd citcoms_assim
make distclean
autoreconf -ivf
module load openmpi
module load hdf5
export LD=ld
./configure
make
```

At the University of Bern you can use the following MPI distribution:

```module load iomkl/2018b```

<!--Python 2 is also necessary for running a script that configures and builds the C code: this script is ```mymake.py```.  You may therefore also need to run ```module load python2``` to provide a python2 distribution.   

```
module load openmpi
module load python2
cd src/
./mymake.py
```
-->

To run jobs, you will need to setup a job submission script and examples are provided in ```jobsubmit/```.


### Quick start for Mac OSX

The following instructions assume you have MacPorts (https://www.macports.org) installed, but you can also use a different package manager (e.g. Homebrew) or install the prerequisite software from source.   

#### 1. Install build tools:

These commands are for MacPorts:

```sudo port install automake autoconf libtool```

<!--Note that ```libtool``` already exists on Mac OSX although it is not the GNU version.  Therefore, MacPorts uses a program name transform such that you must replace ```libtoolize``` in ```mymake.py``` with ```glibtoolize```.-->

#### 2. Install an MPI distribution

#### 2a. MPI option 1: Install Open MPI

```
sudo port install openmpi
sudo port select --set mpi openmpi-mp-fortran
```

#### 2b. MPI option 2: Install MPICH

```
sudo port install mpich
sudo port select --set mpi mpich-mp-fortran
```

#### 3. Obtain code and build

```
git clone https://github.com/EarthByte/citcoms.git citcoms_assim
cd citcoms_assim
make distclean
autoreconf -ivf
./configure
make
```

#### 3. Run

An example command to run a uniprocessor job is:

```mpirun -np 1 CitcomSRegional input.sample```

#### Developer notes

1. I found that a popular debugger (LLDB) seems to prefer Open MPI, whereas a memory management tool (valgrind) prefers MPICH.  Therefore I installed both MPI distributions and switch between them using

        sudo port select --set mpi
2. The Valgrind development cycle is always lagging behind Mac, so Valgrind will probably not work on your latest Mac.  However, you can try downloading a version from the website or trying the development version:

        sudo port install valgrind-devel
3. LLDB (debugger) comes with Xcode and is therefore already available on your Mac.
4. I did not have success compiling with HDF5 support, which may be related to the fact that HDF5 is configured in an unsupported 'Experimental' mode when installed as a variant using MacPorts:

        sudo port install openmpi
        sudo port install hdf5 +openmpi
        sudo port select --set mpi openmpi-mp-fortran

    This returns the message:
    
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        hdf5 will been configured in an unsupported "Experimental" mode due to selected variants. See "port variants hdf5 | grep EXPERIMENTAL" for more information.
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    And during the configuration of CitcomS:
    
        configure: WARNING: header 'hdf5.h' not found; disabling HDF5 support
        
    I did not investigate futher.


## Examples and User Guide

Example input and output configuration files are provided in```examples``` (there is also a ```src/examples``` directory, but these are the original examples provided by CitcomS without the addition of data assimilation examples).  A simple global model to first run is ```examples/Full```.

A user guide is available at ```docs/user_guide```, although some of this content relates to the original pyre version of the code that has been superceded by this version.  Hence some of the parameter names have changed.  Please refer to the code itself and/or this README.md to confirm the parameter names.  An ongoing project is to update this user guide.

## Code Modifications and Parameter Names

The user guide contains information about the input parameters, both for the C code and also the pre- and post-processing python scripts.  Some options in the user guide might be outdated, so prioritise the options in this README.

### Slab and lithosphere assimilation
These parts of the code are commented with 'DJB SLAB'

1. ```lith_age_depth_function``` (bool)
1. ```lith_age_exponent``` (double)
1. ```lith_age_min``` (double)
1. ```lith_age_stencil_value``` (double)
1. ```slab_assim``` (bool)
1. ```slab_assim_file``` (char)
1. ```sten_temp``` as an output option (char)
1. ```internal_vbcs_file``` (bool)
1. ```velocity_internal_file``` (char)
1. ```sten_velo``` as an output option (char)

### Composition
These parts of the code are commented with 'DJB COMP'

1. ```hybrid_method``` (bool)
1. increased memory for tracer arrays (```icushion``` parameter)
1. turn off tracer warnings using ```itracer_warnings=off``` in input cfg file

### Viscosity structures
These parts of the code are commented with 'DJB VISC'

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

### Time output
These parts of the code are commented with 'DJB TIME'

1. output data in regular increments of age (Myr) as well as/rather than number of time steps
    - ```storage_spacing_Myr``` (int)
    - if you only want to output data by age (Ma), you should set ```storage_spacing``` to a large integer value in order to suppress the regular time outputs
    - both ```storage_spacing_Myr``` and ```storage_spacing``` can be used together, in which case data is output whenever either one of these output criterion is satisfied
1. exit time loop when the model reaches negative ages (currently hard-coded to be <0 Ma)
    - ```exit_at_present``` (bool)
    
### Extended-Boussinesq modifications
These parts of the code are commented with 'DJB EBA'

1. depth-dependent scaling for the dissipation number (Di)
    - Di is typically defined at the surface, using surface values of alpha, cp, and g
    - a depth-dependent scaling (typically between 0 and 1) is introduced to additionally scale Di
    - this scales Di in the energy equation, i.e. the adiabatic, viscous, and latent heating terms
    - useful to avoid intense shear heating near the surface and hence avoid artefacts when using data assimilation  
2. Therefore, an extra column is added to the ```refstate_file```:
    - [column 1] rho
    - [column 2] gravity
    - [column 3] thermal expansion coefficient (alpha)
    - [column 4] heat capacity
    - [column 5] dissipation number scaling

### Output fields
These parts of the code are commented with 'DJB OUT'

1. Composition and temperature spherical harmonics as an output option (char).  See issue, since output for comp only occurs for the last comp field.
    - ```comp_sph```
    - ```temp_sph```
1. Temperature and internal velocity stencil for slab assimilation
    - ```sten_temp```
    - ```sten_velo```
1. Tracer density at the nodes (originally written by Ting Yang)
    - ```tracer_dens```
1. Divergence (div/v) at the nodes.  Useful for debugging convergence issues.
    - ```divv```
1. Fixed pid file output for ```lith_age_min```, ```z_interface```, and ```output_optional```
1. Fixed various valgrind uninitialised variable warnings relating to the writing of the pid file

### Topography
These parts of the code are commented with 'DJB TOPO'

1. Parameter to remove buoyancy above a given znode for computing dynamic topography
    - ```remove_buoyancy_above_znode``` (int)

### Ultra-low velocity zone
These parts of the code are commented with 'DJB ULVZ'.  These amendments will not affect data assimilation, but are included for completeness.

1. Modifications to enable ULVZ modelling as in Bower et al. (2011).
    - domain volumes for absolute tracer method for different initial ULVZ volumes
    - permeable domain for regional models
    - sine perturbation for initial temperature

## Affiliated codes

A plume detection code from Rakib Hassan's thesis work is available at
https://github.com/rh-downunder/plume-tracker

<!--
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

1. reverse gravity acceleraton (added by Ting, see svn r76) for SBI
   note that this appears to have been subsequently removed in r88?

### Log of features implemented from legacy code
1. Adv\_diff -> COMPLETE
1. BC\_util -> COMPLETE
1. Composition\_related.c -> COMPLETE
1. composition\_related.h -> COMPLETE
1. Convection.c -> COMPLETE
1. Element\_calculations -> COMPLETE
1. Full\_boundary\_conditions -> COMPLETE
1. Full\_lith\_age\_read\_files -> COMPLETE
1. Full_read_input_from_file -> COMPLETE
1. Full\_solver.c -> COMPLETE
1. global\_defs.h -> COMPLETE
1. Instructions.c -> COMPLETE
1. Lith\_age.c -> COMPLETE
1. output.c -> COMPLETE
1. Pan\_problem.c -> COMPLETE
1. Problem\_related.c -> COMPLETE
1. Regional\_boundary\_conditions.c -> COMPLETE
1. Regional\_lith\_age\_read\_files -> COMPLETE
1. Regional\_read\_input\_files -> COMPLETE
1. Regional\_solver -> COMPLETE
1. solver.h -> COMPLETE
1. Stokes\_flow\_incom -> divv calculation for output COMPLETE
1. tracer\_defs.h -> COMPLETE
1. Viscosity\_structure.c -> COMPLETE
-->

## FAQ
1. **I receive 'Warning: Solver not converging!' when I run a model** This is because the Stokes solver is struggling to compute a velocity and pressure solution that is compatible with your chosen tolerances.  There are several approaches to debugging and rectifying your problem:
    - Most users set tolerances within the range of 1.0E-4 and 5.0E-2 and additionally set ```check_continuity_convergence=off``` (satisfying the tolerance on continuity is typically the most challenging, and many users choose to monitor this residual manually by checking the log and stdout).
    - Reduce the total viscosity contrast in your model and limit the irregularities in viscosity structure.
    - Turn off prescribed velocity boundary conditions and see if you still receive a warning.  Generally, prescribing velocity boundary conditions presents challenges to any Stokes solver (http://lists.geodynamics.org/pipermail/cig-mc/2016-March/000699.html). Ultimately, systematically disentangling the different components of your model is the best way to identify the potential cause of the problem.
    - If you are using GPlates to export your surface velocities, increase the velocity smoothing (in GPlates) to ensure there are no abrupt jumps in surface velocity across plate boundaries.
    - The [augmented Lagrangian number](http://en.wikipedia.org/wiki/Augmented_Lagrangian_method) is by default set to 2E3, but increasing this value by  1 to 2 orders of magnitude may help with the convergence of models with large viscosity contrast .
    - Finally, you could look to tweak some of the other solver parameters, notably relating to the multigrid solver.  But this obviously requires some knowledge of multigrid solvers and how to optimise solvers.
    - Clearly, if you find optimal solver parameters for data assimilation models, please let me know so I can update this FAQ!
    
1. **Why isn't the data assimilation method included in the master version of CitcomS hosted by CIG (https://geodynamics.org)?** To implement data assimilation required changing some functions related to boundary conditions, such that (probably) some of the original functionality is broken (at least for regional models).  Whilst I tried to maintain backwards compatability as much as possible, without a comprehensive series of tests I cannot guarantee that some original functionality has not been disabled.  Furthermore, data assimilation requires pre-processing (python) scripts to generate input data, which are not required for standard CitcomS runs.  Therefore, it was decided to host both the C code and python scripts together in this repository.

1. **How do I start a model?** To start a model you will typically read in an initial temperature field from a velo file, and therefore you set ```tic_method=-1``` where the suffix of the velo file is specified by ```solution_cycles_init```.  You then set the ```start_age``` in Ma and it is also good practice to set ```zero_elapsed_time=1```.  The ```zero_elapsed_time=1``` option ensures that the model age begins at ```start_age```.  However, if you generate the initial temperature field using the assimilation preprocessing scripts, then the elapsed ime is already set to zero in the velocity file header, and therefore ```zero_elapsed_time=1``` is technically not necessary since the header is parsed by CitcomS when ```tic_method=-1```
