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
	BoundedBox.cc \
	BoundedMesh.cc \
	Dimensional.cc \
	Interior.cc \
	InteriorImposing.cc \
	Interpolator.cc \
	Sink.cc \
	Source.cc \
	TractionInterpolator.cc \
	TractionSource.cc \
	bindings.cc \
	exceptions.cc \
	exchangers.cc \
	initTemperature.cc \
	misc.cc \
	utility.cc


# version
# $Id: Make.mm,v 1.10 2003/12/30 21:44:32 tan2 Exp $

# End of file
