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

CC = $(MPICC)
INCLUDES = $(MPIINCLUDES)
lib_LTLIBRARIES = libCitcomSCommon.la
libCitcomSCommon_la_SOURCES = \
	Advection_diffusion.c \
	advection_diffusion.h \
	advection.h \
	Citcom_init.c \
	citcom_init.h \
	Construct_arrays.c \
	Convection.c \
	convection_variables.h \
	Drive_solvers.c \
	drive_solvers.h \
	Element_calculations.c \
	element_definitions.h \
	General_matrix_functions.c \
	global_defs.h \
	Global_operations.c \
	Initial_temperature.c \
	initial_temperature.h \
	Instructions.c \
	Interuption.c \
	interuption.h \
	lith_age.h \
	Nodal_mesh.c \
	Output.c \
	output.h \
	Pan_problem_misc_functions.c \
	parallel_related.h \
	Parsing.c \
	parsing.h \
	Phase_change.c \
	phase_change.h \
	Problem_related.c \
	Process_buoyancy.c \
	Shape_functions.c \
	Size_does_matter.c \
	Solver_conj_grad.c \
	Solver_multigrid.c \
	sphere_communication.h \
	Sphere_harmonics.c \
	Stokes_flow_Incomp.c \
	temperature_descriptions.h \
	Topo_gravity.c \
	Tracer_advection.c \
	tracer_defs.h \
	viscosity_descriptions.h \
	Viscosity_structures.c
libCitcomSCommon_la_LIBADD = $(LIBM)
libCitcomSCommon_la_LDFLAGS = -release $(VERSION)
EXTRA_DIST = \
	Obsolete.c

## end of Makefile.am
