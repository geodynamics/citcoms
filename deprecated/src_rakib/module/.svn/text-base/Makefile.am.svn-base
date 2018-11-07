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

if COND_EXCHANGER
    MAYBE_EXCHANGER = Exchanger
endif

SUBDIRS = $(MAYBE_EXCHANGER)

noinst_LIBRARIES =
pyexec_LTLIBRARIES =

if COND_EMBEDDING
    # static library
    noinst_LIBRARIES += libCitcomSLibmodule.a
else
    # extension module (libtool)
    pyexec_LTLIBRARIES += CitcomSLibmodule.la
endif

AM_CPPFLAGS =
if COND_HDF5
    AM_CPPFLAGS += -DUSE_HDF5
endif

INCLUDES = \
	-I$(top_srcdir)/lib \
	$(PYTHON_EGG_CPPFLAGS) \
	-I$(PYTHON_INCDIR)

# static libraries

libCitcomSLibmodule_a_CFLAGS = $(AM_CFLAGS) # hack for automake
libCitcomSLibmodule_a_SOURCES = $(sources)

# extension modules (libtool)

CitcomSLibmodule_la_LDFLAGS = -module -release $(VERSION)
CitcomSLibmodule_la_LIBADD = \
	$(top_builddir)/lib/libCitcomS.la \
	$(PYTHON_EGG_LDFLAGS)
CitcomSLibmodule_la_SOURCES = $(sources)

# sources

sources = \
	CitcomSmodule.c \
	CitcomSmodule.h \
	advdiffu.c \
	advdiffu.h \
	bindings.c \
	bindings.h \
	exceptions.c \
	exceptions.h \
	getProperty.h \
	initial_conditions.c \
	initial_conditions.h \
	mesher.c \
	mesher.h \
	misc.c \
	misc.h \
	outputs.c \
	outputs.h \
	setProperties.c \
	setProperties.h \
	stokes_solver.c \
	stokes_solver.h

## end of Makefile.am
