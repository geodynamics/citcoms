# -*- Makefile -*-
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                                  Steve Quenette
#                        California Institute of Technology
#                        (C) 1998-2003  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# version
# $Id: Make.mm,v 1.4 2003/04/25 22:29:56 tan2 Exp $

include local.def
TYPE=Regional

PROJECT = CitcomS
PACKAGE = lib/$(TYPE)

PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/$(PACKAGE)
PROJ_LIB = $(BLD_LIBDIR)/libCitcomS$(TYPE).$(EXT_LIB)

PROJ_LIBRARIES = $(EXTERNAL_LIBPATH) $(EXTERNAL_LIBS) -lm
PROJ_CC_INCLUDES = ../Common

PROJ_SRCS = \
	Boundary_conditions.c \
	Geometry_cartesian.c \
	Global_operations.c \
	Initial_temperature.c \
	Lith_age.c \
	Output.c \
	Parallel_related.c \
	Problem_related.c \
	Process_buoyancy.c \
	Sphere_related.c \
	Version_dependent.c

#EXPORT_HEADERS = \
#	global_defs.h

#PROJ_INCDIR = $(BLD_INCDIR)/$(PROJECT)/$(TYPE)
PROJ_CLEAN = $(PROJ_OBJS) $(PROJ_DEPENDENCIES)

#all: $(PROJ_LIB) export-headers
all: $(PROJ_LIB)


# version
# $Id: Make.mm,v 1.4 2003/04/25 22:29:56 tan2 Exp $

#
# End of file
