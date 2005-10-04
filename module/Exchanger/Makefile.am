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

pkgpyexec_LTLIBRARIES = Exchangermodule.la
CXX = $(MPICXX)
CXXLD = @CXX@
INCLUDES = \
	-I$(top_srcdir)/lib/Common \
	-I$(PYTHON_INCDIR) \
	$(MPIINCLUDES)
Exchangermodule_la_SOURCES = \
	AreaWeightedNormal.cc \
	AreaWeightedNormal.h \
	bindings.cc \
	bindings.h \
	Boundary.cc \
	Boundary.h \
	CitcomInterpolator.cc \
	CitcomInterpolator.h \
	CitcomSource.cc \
	CitcomSource.h \
	Convertor.cc \
	Convertor.h \
	exceptions.cc \
	exceptions.h \
	Exchangermodule.cc \
	exchangers.cc \
	exchangers.h \
	global_bbox.cc \
	global_bbox.h \
	initTemperature.cc \
	initTemperature.h \
	inlets_outlets.cc \
	inlets_outlets.h \
	Interior.cc \
	Interior.h \
	misc.cc \
	misc.h \
	SInlet.cc \
	SInlet.h \
	SIUnit.cc \
	SIUnit.h \
	SVTInlet.cc \
	SVTInlet.h \
	SVTOutlet.cc \
	SVTOutlet.h \
	TInlet.cc \
	TInlet.h \
	TOutlet.cc \
	TOutlet.h \
	VOutlet.cc \
	VOutlet.h \
	VTInlet.cc \
	VTInlet.h \
	VTOutlet.cc \
	VTOutlet.h
EXTRA_DIST = \
	BoundaryVTInlet.cc \
	BoundaryVTInlet.h
Exchangermodule_la_LDFLAGS = -module -release $(VERSION)
Exchangermodule_la_LIBADD = \
	$(top_builddir)/lib/Regional/libCitcomSRegional.la \
	-lExchanger \
	-l_mpimodule -ljournal

## end of Makefile.am
