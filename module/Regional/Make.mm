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
    setProperties.cc

# version
# $Id: Make.mm,v 1.10 2003/06/27 00:15:02 tan2 Exp $

# End of file
