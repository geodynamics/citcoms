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
	-Xlinker -rpath $(PROJ_LIBDIR) -Xlinker \
	-l$(PROJECT)Common \
	-l$(PROJECT)$(TYPE) \
	-Xlinker -rpath $(PYTHIA_LIBDIR) -Xlinker \
	-ljournal \
	$(PYTHIA_DIR)/modules/mpi/_mpimodule.so

PROJ_CXX_INCLUDES = ../../lib/Common
EXTERNAL_INCLUDES += $(PYTHIA_DIR)/include $(PYTHIA_INCDIR)
EXTERNAL_LIBPATH += -L$(PYTHIA_DIR)/lib -L$(PYTHIA_LIBDIR)

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
# $Id: Make.mm,v 1.16 2005/06/03 21:51:42 leif Exp $

# End of file
