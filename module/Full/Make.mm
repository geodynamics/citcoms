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
EXTERNAL_LIBPATH += -L$(TOOLS_DIR)/lib

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
# $Id: Make.mm,v 1.9 2003/09/30 00:39:30 tan2 Exp $

# End of file
