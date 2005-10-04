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

pkgpyexec_LTLIBRARIES = Regionalmodule.la
noinst_LTLIBRARIES = libModuleCommon.la
CXX = $(MPICXX)
CXXLD = @CXX@
INCLUDES = \
	-I$(top_srcdir)/lib/Common \
	-I$(PYTHON_INCDIR) \
	$(MPIINCLUDES)

# libModuleCommon
libModuleCommon_la_SOURCES = \
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

# Regionalmodule
Regionalmodule_la_SOURCES = Regionalmodule.cc
Regionalmodule_la_LDFLAGS = -module -release $(VERSION)
Regionalmodule_la_LIBADD = \
	libModuleCommon.la \
	$(top_builddir)/lib/Regional/libCitcomSRegional.la \
	-l_mpimodule -ljournal

## end of Makefile.am
