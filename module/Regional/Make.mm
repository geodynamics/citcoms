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

PROJ_CXX_INCLUDES = ../../lib/Common ../../lib/$(TYPE)

PROJ_SRCS = \
    advdiffu.cc \
    bindings.cc \
    exceptions.cc \
    mesher.cc \
    misc.cc \
    outputs.cc \
    setProperties.cc \
    stokes_solver.cc

# version
# $Id: Make.mm,v 1.14 2003/08/13 01:22:21 tan2 Exp $

# End of file
