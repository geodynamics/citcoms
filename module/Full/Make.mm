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

PROJ_CXX_SRCLIB = \
	-l$(PROJECT)Common \
	-l$(PROJECT)$(TYPE) \
	-ljournal \
	-lmpimodule

PROJ_CXX_INCLUDES = ../../lib/Common
EXTERNAL_LIBPATH += -L$(TOOLS_LIBDIR)

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


#===========================================================================
# link source files from ../Regional to .

link:
	(cd ../Regional;\
	 ln -f $(PROJ_SRCS) *.h ../$(TYPE))

# version
# $Id: Make.mm,v 1.11 2003/11/23 08:31:50 steve Exp $

# End of file
