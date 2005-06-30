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

EXTERNAL_INCLUDES += $(PYTHIA_INCDIR) $(EXCHANGER_INCDIR)
EXTERNAL_LIBDIRS = $(EXCHANGER_LIBDIR) $(PYTHIA_LIBDIR)
EXTERNAL_LIBPATH += $(foreach dir,$(EXTERNAL_LIBDIRS),-L$(dir))
ifeq (Linux,$(findstring Linux,$(PLATFORM_ID)))
RPATH_ARGS = $(foreach dir,$(EXTERNAL_LIBDIRS),-Xlinker -rpath $(dir))
else
RRPATH_ARGS =
endif

PROJ_CXX_SRCLIB = \
	-lRegionalmodule \
	-lExchanger \
	-l_mpimodule \
	-ljournal \
	$(RPATH_ARGS)

PROJ_CXX_INCLUDES = ../../lib/Common

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
# $Id: Make.mm,v 1.30 2005/06/30 01:31:29 leif Exp $

# End of file
