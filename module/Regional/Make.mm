# -*- Makefile -*-
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2003  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

include local.def

TYPE=Regional

PROJECT = CitcomS
PACKAGE = $(PROJECT)$(TYPE)module
include std-pythonmodule.def

PROJ_CXX_SRCLIB = $(BLD_LIBDIR)/lib$(PROJECT)$(TYPE).$(EXT_LIB) $(BLD_LIBDIR)/lib$(PROJECT)$(TYPE)Common.$(EXT_LIB)

PROJ_SRCS = \
    bindings.cc \
    exceptions.cc \
    misc.cc

# version
# $Id: Make.mm,v 1.1 2003/03/24 01:46:37 tan2 Exp $

# End of file
