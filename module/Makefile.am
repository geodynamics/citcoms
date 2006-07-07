## Process this file with automake to produce Makefile.in
##
##<LicenseText>
##
## CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
## Copyright (C) 2002-2005, California Institute of Technology.
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

SUBDIRS = Exchanger

noinst_LIBRARIES =
pkgpyexec_LTLIBRARIES =

if COND_EMBEDDING
    # static library
    noinst_LIBRARIES += libCitcomSmodule.a libPyxMPImodule.a
else
    # extension module (libtool)
    pkgpyexec_LTLIBRARIES += CitcomSmodule.la PyxMPImodule.la
endif

# static libraries

libCitcomSmodule_a_CXXFLAGS = $(AM_CXXFLAGS) # hack for automake
libCitcomSmodule_a_SOURCES = $(sources)

libPyxMPImodule_a_CFLAGS = $(AM_CXXFLAGS) # hack for automake
libPyxMPImodule_a_SOURCES = PyxMPI/PyxMPI.c

# extension modules (libtool)

CitcomSmodule_la_LDFLAGS = -module -release $(VERSION)
CitcomSmodule_la_LIBADD = \
	$(top_builddir)/lib/libCitcomS.la \
	-l_mpimodule -ljournal $(MPILIBS)
CitcomSmodule_la_SOURCES = $(sources)

PyxMPImodule_la_LDFLAGS = -module -release $(VERSION)
PyxMPImodule_la_LIBADD =
PyxMPImodule_la_SOURCES = PyxMPI/PyxMPI.c

# compiler settings

CC = $(MPICC)
CXX = $(MPICXX)
INCLUDES = \
	-I$(top_srcdir)/module/Regional \
	-I$(top_srcdir)/lib \
	-I$(PYTHON_INCDIR) \
	$(MPIINCLUDES)

# sources

sources = \
	CitcomSmodule.cc \
	Regional/advdiffu.cc \
	Regional/advdiffu.h \
	Regional/bindings.cc \
	Regional/bindings.h \
	Regional/exceptions.cc \
	Regional/exceptions.h \
	Regional/initial_conditions.cc \
	Regional/initial_conditions.h \
	Regional/mesher.cc \
	Regional/mesher.h \
	Regional/misc.cc \
	Regional/misc.h \
	Regional/outputs.cc \
	Regional/outputs.h \
	Regional/setProperties.cc \
	Regional/setProperties.h \
	Regional/stokes_solver.cc \
	Regional/stokes_solver.h

## end of Makefile.am
