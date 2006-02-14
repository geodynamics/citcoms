# -*- Makefile -*-
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

PROJECT = tabulator
PACKAGE = _tabulatormodule
MODULE = _tabulator

include std-pythonmodule.def
include local.def

PROJ_CXX_SRCLIB = -ltabulator $(COMPILER_LCXX_FORTRAN)

PROJ_SRCS = \
    bindings.cc \
    exceptions.cc \
    misc.cc \
    tabulator.cc


# version
# $Id: Make.mm,v 1.1.1.1 2005/03/17 20:03:02 aivazis Exp $

# End of file
