# -*- Makefile -*-
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
	Boundary.cc \
	CoarseGridExchanger.cc \
	ExchangerClass.cc \
	FineGridExchanger.cc \
	bindings.cc \
	exceptions.cc \
	exchangers.cc \
	misc.cc


# version
# $Id: Make.mm,v 1.4 2003/10/02 01:14:22 tan2 Exp $

# End of file
