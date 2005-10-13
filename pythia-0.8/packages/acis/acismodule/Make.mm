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

PROJECT = acis

include std-pythonmodule.def
include local.def

PROJ_CXX_SRCLIB = -ljournal

PROJ_SRCS = \
    Mesher.cc \
    attributes.cc \
    bindings.cc \
    debug.cc \
    entities.cc \
    exceptions.cc \
    faceting.cc \
    intersections.cc \
    misc.cc \
    solids.cc \
    operators.cc \
    transformations.cc \
    support.cc \
    util.cc

ifdef $(ACIS_HAS_MESHER)
    PROJ_SRCS += meshing.cc
endif

# version
# $Id: Make.mm,v 1.1.1.1 2005/03/08 16:13:32 aivazis Exp $

# End of file
