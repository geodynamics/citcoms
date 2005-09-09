## Process this file with automake to produce Makefile.in
# $Id: Makefile.am,v 1.1 2005/08/24 22:45:51 leif Exp $

CC = $(MPICC)
INCLUDES = -I$(top_srcdir)/lib/Common $(MPIINCLUDES)
lib_LTLIBRARIES = libCitcomSRegional.la
libCitcomSRegional_la_SOURCES = \
	Boundary_conditions.c \
	Geometry_cartesian.c \
	Lith_age.c \
	Parallel_related.c \
	Read_input_from_files.c \
	Sphere_related.c \
	Version_dependent.c
libCitcomSRegional_la_LIBADD = \
	$(top_builddir)/lib/Common/libCitcomSCommon.la \
	$(LIBM)
libCitcomSRegional_la_LDFLAGS = -release $(VERSION)
EXTRA_DIST = \
	Obsolete.c

## end of Makefile.am
