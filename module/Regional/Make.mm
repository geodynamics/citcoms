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

PROJ_CXX_SRCLIB = $(BLD_LIBDIR)/lib$(PROJECT)$(TYPE).$(EXT_LIB) 
PROJ_CXX_INCLUDES = ../../lib/Common ../../lib/$(TYPE)
#PROJ_CXX_INCLUDES = $(BLD_INCDIR)/$(PROJECT)/$(TYPE)

PROJ_SRCS = \
    bindings.cc \
    exceptions.cc \
    misc.cc

# version
# $Id: Make.mm,v 1.3 2003/04/05 20:25:10 tan2 Exp $

# End of file
