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
PROJ_CC_INCLUDES = ../$(TYPE) .

PROJ_SRCS = \
	Advection_diffusion.c \
	Citcom_init.c \
	Construct_arrays.c \
	Convection.c \
	Drive_solvers.c \
	Element_calculations.c \
	General_matrix_functions.c \
	Global_operations.c \
	Instructions.c \
	Interuption.c \
	Nodal_mesh.c \
	Pan_problem_misc_functions.c \
	Parsing.c \
	Phase_change.c \
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
	advection.h \
	citcom_init.h \
	convection_variables.h \
	element_definitions.h \
	sphere_communication.h \
	temperature_descriptions.h \
	tracer_defs.h \
	viscosity_descriptions.h

PROJ_INCDIR = $(BLD_INCDIR)/$(PROJECT)/$(TYPE)
PROJ_CLEAN = $(PROJ_OBJS) $(PROJ_DEPENDENCIES)

#all: $(PROJ_OBJS) export-headers
all: $(PROJ_OBJS)

# version
# $Id: Make.mm,v 1.7 2003/08/06 18:47:45 tan2 Exp $

#
# End of file
