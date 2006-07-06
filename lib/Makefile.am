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

lib_LTLIBRARIES = libCitcomS.la
noinst_LTLIBRARIES = libFull.la libRegional.la

CC = $(MPICC)
INCLUDES = -I$(top_srcdir)/lib/Common $(MPIINCLUDES)

libCitcomS_la_SOURCES = \
	Common/Advection_diffusion.c \
	Common/advection_diffusion.h \
	Common/advection.h \
	Common/BC_util.c \
	Common/Citcom_init.c \
	Common/citcom_init.h \
	Common/Construct_arrays.c \
	Common/Convection.c \
	Common/convection_variables.h \
	Common/Drive_solvers.c \
	Common/drive_solvers.h \
	Common/Element_calculations.c \
	Common/element_definitions.h \
	Common/General_matrix_functions.c \
	Common/global_defs.h \
	Common/Global_operations.c \
	Common/Initial_temperature.c \
	Common/initial_temperature.h \
	Common/Instructions.c \
	Common/Interuption.c \
	Common/interuption.h \
	Common/lith_age.h \
	Common/Lith_age.c \
	Common/Nodal_mesh.c \
	Common/Output.c \
	Common/output.h \
	Common/Pan_problem_misc_functions.c \
	Common/parallel_related.h \
	Common/Parallel_util.c \
	Common/Parsing.c \
	Common/parsing.h \
	Common/Phase_change.c \
	Common/phase_change.h \
	Common/Problem_related.c \
	Common/Process_buoyancy.c \
	Common/Shape_functions.c \
	Common/Size_does_matter.c \
	Common/Solver_conj_grad.c \
	Common/Solver_multigrid.c \
	Common/sphere_communication.h \
	Common/Sphere_harmonics.c \
	Common/Sphere_util.c \
	Common/Stokes_flow_Incomp.c \
	Common/temperature_descriptions.h \
	Common/Topo_gravity.c \
	Common/Tracer_advection.c \
	Common/tracer_defs.h \
	Common/viscosity_descriptions.h \
	Common/Viscosity_structures.c
libCitcomS_la_LIBADD = libFull.la libRegional.la $(LIBM)
libCitcomS_la_LDFLAGS = -release $(VERSION)

libFull_la_CFLAGS = $(AM_CFLAGS) # hack for automake
libFull_la_SOURCES = \
	Full/Boundary_conditions.c \
	Full/Geometry_cartesian.c \
	Full/Lith_age_read_files.c \
	Full/Parallel_related.c \
	Full/Read_input_from_files.c \
	Full/Solver.c \
	Full/Sphere_related.c \
	Full/Version_dependent.c

libRegional_la_CFLAGS = $(AM_CFLAGS) # hack for automake
libRegional_la_SOURCES = \
	Regional/Boundary_conditions.c \
	Regional/Geometry_cartesian.c \
	Regional/Lith_age_read_files.c \
	Regional/Parallel_related.c \
	Regional/Read_input_from_files.c \
	Regional/Solver.c \
	Regional/Sphere_related.c \
	Regional/Version_dependent.c

EXTRA_DIST = \
	Common/Obsolete.c \
	Full/Obsolete.c \
	Regional/Obsolete.c

## end of Makefile.am
