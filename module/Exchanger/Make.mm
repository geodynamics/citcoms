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
PROJ_CXX_INCLUDES = ../../lib/Common 

PROJ_CXX_SRCLIB = \
        $(EXPORT_ROOT)/modules/$(PROJECT)/Regionalmodule.so \
        -ljournal \
        -lmpimodule

EXTERNAL_LIBPATH += -L$(TOOLS_DIR)/lib

PROJ_SRCS = \
	AreaWeightedNormal.cc \
	Boundary.cc \
	BoundaryCondition.cc \
	BoundaryVTInlet.cc \
	BoundedBox.cc \
	BoundedMesh.cc \
	CartesianCoord.cc \
	Convertor.cc \
	FEMInterpolator.cc \
	Inlet.cc \
	Interior.cc \
	InteriorImposing.cc \
	Outlet.cc \
	SIUnit.cc \
	Sink.cc \
	Source.cc \
	TractionBC.cc \
	TractionInterpolator.cc \
	TractionOutlet.cc \
	TractionSource.cc \
	VTInlet.cc \
	VTInterpolator.cc \
	VTOutlet.cc \
	VTSource.cc \
	bindings.cc \
	exceptions.cc \
	exchangers.cc \
	initTemperature.cc \
	inlets_outlets.cc \
	misc.cc \
	utility.cc


# version
# $Id: Make.mm,v 1.15 2004/02/24 20:40:06 tan2 Exp $

# End of file
