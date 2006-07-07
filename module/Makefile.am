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
pyexec_LTLIBRARIES =

if COND_EMBEDDING
    # static library
    noinst_LIBRARIES += libCitcomSLibmodule.a libPyxMPImodule.a
else
    # extension module (libtool)
    pyexec_LTLIBRARIES += CitcomSLibmodule.la PyxMPImodule.la
endif

# static libraries

libCitcomSLibmodule_a_CXXFLAGS = $(AM_CXXFLAGS) # hack for automake
libCitcomSLibmodule_a_SOURCES = $(sources)

libPyxMPImodule_a_CFLAGS = $(AM_CXXFLAGS) # hack for automake
libPyxMPImodule_a_SOURCES = PyxMPI/PyxMPI.c

# extension modules (libtool)

CitcomSLibmodule_la_LDFLAGS = -module -release $(VERSION)
CitcomSLibmodule_la_LIBADD = \
	$(top_builddir)/lib/libCitcomS.la \
	-l_mpimodule -ljournal $(MPILIBS)
CitcomSLibmodule_la_SOURCES = $(sources)

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
	advdiffu.cc \
	advdiffu.h \
	bindings.cc \
	bindings.h \
	exceptions.cc \
	exceptions.h \
	initial_conditions.cc \
	initial_conditions.h \
	mesher.cc \
	mesher.h \
	misc.cc \
	misc.h \
	outputs.cc \
	outputs.h \
	setProperties.cc \
	setProperties.h \
	stokes_solver.cc \
	stokes_solver.h

## end of Makefile.am
