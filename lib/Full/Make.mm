# -*- Full Makefile -*-
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

include local.def
TYPE=Full

PROJECT = CitcomS
PACKAGE = lib/$(TYPE)

PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/$(PACKAGE)
PROJ_LIB = $(BLD_LIBDIR)/libCitcomS$(TYPE).$(EXT_LIB)

PROJ_LIBRARIES = $(EXTERNAL_LIBPATH) $(EXTERNAL_LIBS) -lm
#PROJ_CC_INCLUDES = ../Common
PROJ_CC_INCLUDES = ./

PROJ_SRCS = \
	Boundary_conditions.c \
	Geometry_cartesian.c \
	Lith_age.c \
	Parallel_related.c \
	Sphere_related.c \
	Version_dependent.c

PROJ_CLEAN = $(PROJ_OBJS) $(PROJ_DEPENDENCIES)



all: $(PROJ_LIB)


# version
# $Id: Make.mm,v 1.10 2003/08/12 16:34:19 ces74 Exp $

#
# End of file
