# -*- Regional Makefile -*-
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

include local.def
TYPE=Regional

PROJECT = CitcomS
PACKAGE = lib/$(TYPE)

PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/$(PACKAGE)
PROJ_LIB = $(BLD_LIBDIR)/libCitcomS$(TYPE).$(EXT_LIB)

PROJ_LIBRARIES = $(EXTERNAL_LIBPATH) $(EXTERNAL_LIBS) -lm
PROJ_CC_INCLUDES = ../Common ./

PROJ_SRCS = \
	Boundary_conditions.c \
	Geometry_cartesian.c \
	Lith_age.c \
	Parallel_related.c \
	Read_input_from_files.c \
	Sphere_related.c \
	Version_dependent.c

PROJ_CLEAN = $(PROJ_OBJS) $(PROJ_DEPENDENCIES)



all: $(PROJ_LIB)


# version
# $Id: Make.mm,v 1.12 2003/09/01 18:30:21 ces74 Exp $

#
# End of file
