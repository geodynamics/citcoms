## Process this file with automake to produce Makefile.in
##
##<LicenseText>
##
## CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
## Copyright (C) 2002-2005, California Institute of Technology.
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

SUBDIRS =
DIST_SUBDIRS = examples tests visual
MERLIN_SUBDIRS = module

bin_PROGRAMS = CitcomSFull CitcomSRegional

CitcomSFull_CPPFLAGS = -I$(top_srcdir)/lib
CitcomSFull_SOURCES = Citcom.c CitcomSFull.c
CitcomSFull_LDADD = libCitcomS.a

CitcomSRegional_CPPFLAGS = -I$(top_srcdir)/lib
CitcomSRegional_SOURCES = Citcom.c CitcomSRegional.c
CitcomSRegional_LDADD = libCitcomS.a

noinst_LIBRARIES = libCitcomS.a

libCitcomS_a_CPPFLAGS = -I$(top_srcdir)/lib
libCitcomS_a_SOURCES = \
	lib/Advection_diffusion.c \
	lib/advection_diffusion.h \
	lib/advection.h \
	lib/BC_util.c \
	lib/Citcom_init.c \
	lib/citcom_init.h \
	lib/Construct_arrays.c \
	lib/Convection.c \
	lib/convection_variables.h \
	lib/Drive_solvers.c \
	lib/drive_solvers.h \
	lib/Element_calculations.c \
	lib/element_definitions.h \
	lib/General_matrix_functions.c \
	lib/global_defs.h \
	lib/Global_operations.c \
	lib/hdf5_related.h \
	lib/Initial_temperature.c \
	lib/initial_temperature.h \
	lib/Instructions.c \
	lib/Interuption.c \
	lib/interuption.h \
	lib/lith_age.h \
	lib/Lith_age.c \
	lib/Nodal_mesh.c \
	lib/Output.c \
	lib/output.h \
	lib/Output_h5.c \
	lib/output_h5.h \
	lib/Pan_problem_misc_functions.c \
	lib/parallel_related.h \
	lib/Parallel_util.c \
	lib/Parsing.c \
	lib/parsing.h \
	lib/Phase_change.c \
	lib/phase_change.h \
	lib/Problem_related.c \
	lib/Process_buoyancy.c \
	lib/Shape_functions.c \
	lib/Size_does_matter.c \
	lib/Solver_conj_grad.c \
	lib/Solver_multigrid.c \
	lib/solver.h \
	lib/sphere_communication.h \
	lib/Sphere_harmonics.c \
	lib/Sphere_util.c \
	lib/Stokes_flow_Incomp.c \
	lib/temperature_descriptions.h \
	lib/Topo_gravity.c \
	lib/Tracer_advection.c \
	lib/tracer_defs.h \
	lib/viscosity_descriptions.h \
	lib/Viscosity_structures.c \
	lib/Full_boundary_conditions.c \
	lib/Full_geometry_cartesian.c \
	lib/Full_lith_age_read_files.c \
	lib/Full_parallel_related.c \
	lib/Full_read_input_from_files.c \
	lib/Full_solver.c \
	lib/Full_sphere_related.c \
	lib/Full_version_dependent.c \
	lib/Regional_boundary_conditions.c \
	lib/Regional_geometry_cartesian.c \
	lib/Regional_lith_age_read_files.c \
	lib/Regional_parallel_related.c \
	lib/Regional_read_input_from_files.c \
	lib/Regional_solver.c \
	lib/Regional_sphere_related.c \
	lib/Regional_version_dependent.c

EXTRA_DIST = \
	lib/Obsolete.c \
	lib/Full_obsolete.c \
	lib/Regional_obsolete.c

## end of Makefile.am
