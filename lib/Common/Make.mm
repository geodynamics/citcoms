# -*- Common Makefile -*-
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#=====================================================================
#
#                              CitcomS
#                 ---------------------------------
#
#                              Authors:
#           Louis Moresi, Shijie Zhong, Lijie Han, Eh Tan,
#           Clint Conrad, Michael Gurnis, and Eun-seo Choi
#          (c) California Institute of Technology 1994-2005
#
#        By downloading and/or installing this software you have
#       agreed to the CitcomS.py-LICENSE bundled with this software.
#             Free for non-commercial academic research ONLY.
#      This program is distributed WITHOUT ANY WARRANTY whatsoever.
#
#=====================================================================
#
#  Copyright June 2005, by the California Institute of Technology.
#  ALL RIGHTS RESERVED. United States Government Sponsorship Acknowledged.
# 
#  Any commercial use must be negotiated with the Office of Technology
#  Transfer at the California Institute of Technology. This software
#  may be subject to U.S. export control laws and regulations. By
#  accepting this software, the user agrees to comply with all
#  applicable U.S. export laws and regulations, including the
#  International Traffic and Arms Regulations, 22 C.F.R. 120-130 and
#  the Export Administration Regulations, 15 C.F.R. 730-744. User has
#  the responsibility to obtain export licenses, or other export
#  authority as may be required before exporting such information to
#  foreign countries or providing access to foreign nationals.  In no
#  event shall the California Institute of Technology be liable to any
#  party for direct, indirect, special, incidental or consequential
#  damages, including lost profits, arising out of the use of this
#  software and its documentation, even if the California Institute of
#  Technology has been advised of the possibility of such damage.
# 
#  The California Institute of Technology specifically disclaims any
#  warranties, including the implied warranties or merchantability and
#  fitness for a particular purpose. The software and documentation
#  provided hereunder is on an "as is" basis, and the California
#  Institute of Technology has no obligations to provide maintenance,
#  support, updates, enhancements or modifications.
#
#=====================================================================
#</LicenseText>
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
# $Id: Make.mm,v 1.15 2005/06/10 02:23:15 leif Exp $

#
# End of file
