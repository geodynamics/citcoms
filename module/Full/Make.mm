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
	$(PROJ_LIBDIR)/lib$(PROJECT)Common.so \
	$(PROJ_LIBDIR)/lib$(PROJECT)$(TYPE).so \
	-ljournal \
	$(PYTHIA_DIR)/modules/mpi/mpimodule.so

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
# $Id: Make.mm,v 1.14 2005/05/17 00:35:10 tan2 Exp $

# End of file
