## Process this file with automake to produce Makefile.in
##
##<LicenseText>
##
## CitcomS by Louis Moresi, Shijie Zhong, Lijie Han, Eh Tan,
## Clint Conrad, Michael Gurnis, and Eun-seo Choi.
## Copyright (C) 1994-2005, California Institute of Technology.
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; if not, write to the Free Software
## Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##
##</LicenseText>

# $Id$

noinst_LIBRARIES =
lib_LTLIBRARIES =

# static library
noinst_LIBRARIES += libCitcomS.a

AM_CPPFLAGS =
if COND_HDF5
    AM_CPPFLAGS += -DUSE_HDF5
endif

CUDA_SOURCES =

if COND_CUDA
AM_CPPFLAGS += -DUSE_CUDA

.cu.o:
	$(CC) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) $(libCitcomS_a_CFLAGS) $(CFLAGS) -c -o $@ $<

CUDA_SOURCES += cgrad_kernel.cu multigrid_kernel.cu
endif

# static library
libCitcomS_a_CFLAGS = $(AM_CFLAGS) # hack for automake
libCitcomS_a_SOURCES = $(sources)

# shared library (libtool)
#libCitcomS_la_LDFLAGS = -release $(VERSION)
#libCitcomS_la_SOURCES = $(sources)

sources = \
	Advection_diffusion.c \
	Anisotropic_viscosity.c \
	advection_diffusion.h \
	anisotropic_viscosity.h \
	advection.h \
	BC_util.c \
	Checkpoints.c \
	checkpoints.h \
	Citcom_init.c \
	citcom_init.h \
	Composition_related.c \
	composition_related.h \
	Construct_arrays.c \
	Convection.c \
	convection_variables.h \
	Determine_net_rotation.c \
	Drive_solvers.c \
	drive_solvers.h \
	Element_calculations.c \
	element_definitions.h \
	General_matrix_functions.c \
	global_defs.h \
	Ggrd_handling.c \
	Global_operations.c \
	hdf5_related.h \
	Initial_temperature.c \
	initial_temperature.h \
	Instructions.c \
	Interuption.c \
	interuption.h \
	lith_age.h \
	Lith_age.c \
	Material_properties.c \
	material_properties.h \
	Mineral_physics_models.c \
	Nodal_mesh.c \
	Output.c \
	output.h \
	Output_gzdir.c \
	Output_h5.c \
	output_h5.h \
	Output_vtk.c \
	Pan_problem_misc_functions.c \
	parallel_related.h \
	Parallel_util.c \
	Parsing.c \
	parsing.h \
	Phase_change.c \
	phase_change.h \
	Problem_related.c \
	Process_buoyancy.c \
	prototypes.h \
	Shape_functions.c \
	Size_does_matter.c \
	Solver_conj_grad.c \
	Solver_multigrid.c \
	solver.h \
	sphere_communication.h \
	Sphere_harmonics.c \
	Sphere_util.c \
	Stokes_flow_Incomp.c \
	Topo_gravity.c \
	tracer_defs.h \
	Tracer_setup.c \
	viscosity_descriptions.h \
	Viscosity_structures.c \
	Full_boundary_conditions.c \
	Full_geometry_cartesian.c \
	Full_lith_age_read_files.c \
	Full_parallel_related.c \
	Full_read_input_from_files.c \
	Full_solver.c \
	Full_sphere_related.c \
	Full_tracer_advection.c \
	Full_version_dependent.c \
	Regional_boundary_conditions.c \
	Regional_geometry_cartesian.c \
	Regional_lith_age_read_files.c \
	Regional_parallel_related.c \
	Regional_read_input_from_files.c \
	Regional_solver.c \
	Regional_sphere_related.c \
	Regional_tracer_advection.c \
	Regional_version_dependent.c \
	$(CUDA_SOURCES)

EXTRA_DIST = \
	Obsolete.c \
	Full_obsolete.c \
	Regional_obsolete.c

## end of Makefile.am
