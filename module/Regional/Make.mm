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

PROJ_CXX_SRCLIB = $(BLD_LIBDIR)/lib$(PROJECT)$(TYPE).$(EXT_LIB) \
                  $(BLD_LIBDIR)/libmpimodule.a
EXTERNAL_LIBS += -ljournal
PROJ_CXX_INCLUDES = ../../lib/Common ../../lib/$(TYPE)

PROJ_SRCS = \
    advdiffu.cc \
    bindings.cc \
    exceptions.cc \
    misc.cc \
    outputs.cc \
    setProperties.cc \
    stokes_solver.cc

# version
# $Id: Make.mm,v 1.11 2003/07/15 00:40:03 tan2 Exp $

# End of file
