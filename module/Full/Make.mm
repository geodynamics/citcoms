# -*- Makefile -*-
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

include local.def
TYPE=Full

PROJECT = CitcomS
PACKAGE = $(TYPE)module
include std-pythonmodule.def

PROJ_CXX_SRCLIB = $(BLD_LIBDIR)/lib$(PROJECT)Common.$(EXT_LIB) \
		$(BLD_LIBDIR)/lib$(PROJECT)$(TYPE).$(EXT_LIB) \
                  $(BLD_LIBDIR)/libmpimodule.a
EXTERNAL_LIBS += -ljournal
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


#===========================================================================
# link source files from ../Regional to .

link:
	(cd ../Regional;\
	 ln -f $(PROJ_SRCS) *.h ../$(TYPE))

# version
# $Id: Make.mm,v 1.5 2003/08/12 16:34:19 ces74 Exp $

# End of file
