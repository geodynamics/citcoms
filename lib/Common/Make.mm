# -*- Common Makefile -*-
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

include local.def
TYPE=Common

PROJECT = CitcomS
PACKAGE = libCitcomS$(TYPE)

PROJ_SAR = $(BLD_LIBDIR)/$(PACKAGE).$(EXT_SAR)
PROJ_DLL = $(BLD_LIBDIR)/$(PACKAGE).$(EXT_SO)
PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/$(PACKAGE)
PROJ_CLEAN += $(PROJ_DLL) $(PROJ_SAR)

PROJ_INCDIR = $(BLD_INCDIR)/$(PROJECT)/$(TYPE)
PROJ_LIBRARIES = $(EXTERNAL_LIBPATH) $(EXTERNAL_LIBS) -lm
PROJ_CC_INCLUDES = ./

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

EXPORT_LIBS = $(PROJ_SAR)
EXPORT_BINS = $(PROJ_DLL)

#al: $(PROJ_OBJS) export-headers
all: $(PROJ_SAR)

# version
# $Id: Make.mm,v 1.14 2005/04/11 12:09:20 steve Exp $

#
# End of file
