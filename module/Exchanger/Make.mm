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
EXTERNAL_INCLUDES += $(PYTHIA_INCDIR) $(EXCHANGER_INCDIR)
EXTERNAL_LIBPATH += -L$(PYTHIA_LIBDIR) -L$(EXCHANGER_LIBDIR)

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
	SInlet.cc \
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
# $Id: Make.mm,v 1.27 2005/03/11 04:23:33 steve Exp $

# End of file
