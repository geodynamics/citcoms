# -*- Makefile -*-
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

PROJECT = CitcomS
PACKAGE = Exchangermodule

include std-pythonmodule.def

PROJ_CXX_SRCLIB = \
        $(EXPORT_ROOT)/modules/$(PROJECT)/Regionalmodule.so \
	-lExchanger \
        -ljournal \
        $(PYTHIA_DIR)/modules/mpi/mpimodule.so

PROJ_CXX_INCLUDES = ../../lib/Common
EXTERNAL_INCLUDES += $(PYTHIA_INCDIR)
EXTERNAL_LIBPATH += -L$(PYTHIA_LIBDIR)

PROJ_SRCS = \
	AreaWeightedNormal.cc \
	Boundary.cc \
	CitcomInterpolator.cc \
	CitcomSource.cc \
	Convertor.cc \
	Interior.cc \
	SIUnit.cc \
	SVTInlet.cc \
	SVTOutlet.cc \
	TInlet.cc \
	TOutlet.cc \
	VTInlet.cc \
	VOutlet.cc \
	VTOutlet.cc \
	bindings.cc \
	global_bbox.cc \
	exceptions.cc \
	exchangers.cc \
	initTemperature.cc \
	inlets_outlets.cc \
	misc.cc \


# version
# $Id: Make.mm,v 1.25 2005/01/28 19:20:28 tan2 Exp $

# End of file
