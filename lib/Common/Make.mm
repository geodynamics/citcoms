# -*- Makefile -*-
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

include local.def

PROJECT = CitcomS
PACKAGE = lib/$(TYPE)

PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/$(PACKAGE)
#PROJ_LIB = $(BLD_LIBDIR)/libCitcomS$(TYPE)Common.$(EXT_LIB)
#PROJ_CC_INCLUDES = .

PROJ_SRCS = \
	Advection_diffusion.c \
	Citcom_init.c \
	Construct_arrays.c \
	Convection.c \
	Drive_solvers.c \
	Element_calculations.c \
	General_matrix_functions.c \
	Global_operations.c \
	Initial_temperature.c \
	Instructions.c \
	Interuption.c \
	Nodal_mesh.c \
	Output.c \
	Pan_problem_misc_functions.c \
	Parsing.c \
	Phase_change.c \
	Problem_related.c \
	Process_buoyancy.c \
	Process_velocity.c \
	Shape_functions.c \
	Size_does_matter.c \
	Solver_conj_grad.c \
	Solver_multigrid.c \
	Sphere_harmonics.c \
	Stokes_flow_Incomp.c \
	Topo_gravity.c \
	Tracer_advection.c \
	Viscosity_structures.c

EXPORT_HEADERS = \
	advection_diffusion.h \
	advection.h \
	citcom_init.h \
	convection_variables.h \
	drive_solvers.h \
	element_definitions.h a \
	global_defs.h \
	initial_temperature.h \
	interuption.h \
	lith_age.h \
	output.h \
	parallel_related.h \
	parsing.h \
	phase_change.h \
	sphere_communication.h \
	temperature_descriptions.h \
	tracer_defs.h \
	viscosity_descriptions.h

PROJ_INCDIR = $(BLD_INCDIR)/$(PROJECT)/$(TYPE)
PROJ_CLEAN = $(PROJ_OBJS) $(PROJ_DEPENDENCIES)

#all: $(PROJ_OBJS) export-headers
all: $(PROJ_OBJS)

# version
# $Id: Make.mm,v 1.11 2003/08/08 22:51:53 tan2 Exp $

#
# End of file
