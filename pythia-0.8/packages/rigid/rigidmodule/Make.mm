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

PROJECT = rigid
PACKAGE = rigidmodule
MODULE = rigid

include std-pythonmodule.def
include local.def

PROJ_CXX_SRCLIB = -ljournal

PROJ_SRCS = \
    bindings.cc \
    driver.cc \
    exceptions.cc \
    misc.cc


# version
# $Id: Make.mm,v 1.1.1.1 2005/03/08 16:13:58 aivazis Exp $

# End of file
