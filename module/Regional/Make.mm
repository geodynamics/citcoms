# -*- Makefile -*-
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

include local.def
TYPE=Regional

PROJECT = CitcomS
PACKAGE = $(TYPE)module
include std-pythonmodule.def

PROJ_CXX_SRCLIB = \
	-l$(PROJECT)Common \
	-l$(PROJECT)$(TYPE) \
	-ljournal \
	-lmpimodule

PROJ_CXX_INCLUDES = ../../lib/Common
EXTERNAL_LIBPATH += -L$(TOOLS_DIR)/lib

PROJ_SRCS = \
    advdiffu.cc \
    bindings.cc \
    exceptions.cc \
    initial_conditions.cc \
    mesher.cc \
    misc.cc \
    outputs.cc \
    setProperties.cc \
    stokes_solver.cc

# version
# $Id: Make.mm,v 1.18 2003/10/29 18:40:00 tan2 Exp $

# End of file
