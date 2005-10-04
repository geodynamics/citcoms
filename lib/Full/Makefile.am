## Process this file with automake to produce Makefile.in
##
##<LicenseText>
##
## CitcomS by Louis Moresi, Shijie Zhong, Lijie Han, Eh Tan,
## Clint Conrad, Michael Gurnis, and Eun-seo Choi.
## Copyright (C) 1994-2005, California Institute of Technology.
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; if not, write to the Free Software
## Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##
##</LicenseText>

# $Id$

CC = $(MPICC)
INCLUDES = -I$(top_srcdir)/lib/Common $(MPIINCLUDES)
lib_LTLIBRARIES = libCitcomSFull.la
libCitcomSFull_la_SOURCES = \
	Boundary_conditions.c \
	Geometry_cartesian.c \
	Lith_age.c \
	Parallel_related.c \
	Read_input_from_files.c \
	Sphere_related.c \
	Version_dependent.c
libCitcomSFull_la_LIBADD = \
	$(top_builddir)/lib/Common/libCitcomSCommon.la \
	$(LIBM)
libCitcomSFull_la_LDFLAGS = -release $(VERSION)
EXTRA_DIST = \
	Obsolete.c

## end of Makefile.am
