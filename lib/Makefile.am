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

if COND_PYRE

if COND_EMBEDDING
    # static library
    noinst_LIBRARIES += libCitcomS.a
else
    # shared library (libtool)
    lib_LTLIBRARIES += libCitcomS.la
endif

else
    # static library
    noinst_LIBRARIES += libCitcomS.a
endif

AM_CPPFLAGS =
if COND_HDF5
    AM_CPPFLAGS += -DUSE_HDF5
endif

# static library
libCitcomS_a_CXXFLAGS = $(AM_CXXFLAGS) # hack for automake
libCitcomS_a_SOURCES = $(sources)

# shared library (libtool)
libCitcomS_la_LDFLAGS = -release $(VERSION)
libCitcomS_la_SOURCES = $(sources)

sources = \
	Advection_diffusion.cc \
	advection_diffusion.h \
	advection.h \
	BC_util.cc \
	Checkpoints.cc \
	checkpoints.h \
	Citcom_init.cc \
	citcom_init.h \
	Composition_related.cc \
	composition_related.h \
	Construct_arrays.cc \
	Convection.cc \
	convection_variables.h \
	Determine_net_rotation.cc \
	Drive_solvers.cc \
	drive_solvers.h \
	Element_calculations.cc \
	element_definitions.h \
	General_matrix_functions.cc \
	global_defs.h \
	Ggrd_handling.cc \
	Global_operations.cc \
	hdf5_related.h \
	Initial_temperature.cc \
	initial_temperature.h \
	Instructions.cc \
	Interuption.cc \
	interuption.h \
	lith_age.h \
	Lith_age.cc \
	Material_properties.cc \
	material_properties.h \
	Nodal_mesh.cc \
	Output.cc \
	output.h \
	Output_gzdir.cc \
	Output_h5.cc \
	output_h5.h \
	Output_vtk.cc \
	Pan_problem_misc_functions.cc \
	parallel_related.h \
	Parallel_util.cc \
	Parsing.cc \
	parsing.h \
	Phase_change.cc \
	phase_change.h \
	Problem_related.cc \
	Process_buoyancy.cc \
	Shape_functions.cc \
	Size_does_matter.cc \
	Solver_conj_grad.cc \
	Solver_multigrid.cc \
	solver.h \
	sphere_communication.h \
	Sphere_harmonics.cc \
	Sphere_util.cc \
	Stokes_flow_Incomp.cc \
	Topo_gravity.cc \
	tracer_defs.h \
	Tracer_setup.cc \
	viscosity_descriptions.h \
	Viscosity_structures.cc \
	Full_boundary_conditions.cc \
	Full_geometry_cartesian.cc \
	Full_lith_age_read_files.cc \
	Full_parallel_related.cc \
	Full_read_input_from_files.cc \
	Full_solver.cc \
	Full_sphere_related.cc \
	Full_tracer_advection.cc \
	Full_version_dependent.cc \
	Regional_boundary_conditions.cc \
	Regional_geometry_cartesian.cc \
	Regional_lith_age_read_files.cc \
	Regional_parallel_related.cc \
	Regional_read_input_from_files.cc \
	Regional_solver.cc \
	Regional_sphere_related.cc \
	Regional_tracer_advection.cc \
	Regional_version_dependent.cc

EXTRA_DIST = \
	Obsolete.cc \
	Full_obsolete.cc \
	Regional_obsolete.cc

## end of Makefile.am
