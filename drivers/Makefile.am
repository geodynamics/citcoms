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

bin_PROGRAMS = CitcomSFull CitcomSRegional
nodist_bin_SCRIPTS =

if COND_PYRE

nodist_bin_SCRIPTS += citcoms

if COND_EMBEDDING
    bin_PROGRAMS += pycitcoms
    INTERPRETER = $(bindir)/pycitcoms
else
    INTERPRETER = $(PYTHON)
endif

endif

CC = $(MPICC)
CXX = $(MPICXX)
INCLUDES = \
	-I$(top_srcdir)/lib/Common \
	$(MPIINCLUDES)

if COND_PYRE
INCLUDES += \
	-I$(top_srcdir)/module \
	-I$(top_srcdir)/module/Exchanger \
	-I$(PYTHON_INCDIR)
endif

# legacy drivers
CitcomSFull_SOURCES = Citcom.c
CitcomSFull_CPPFLAGS = -DCITCOMS_SOLVER_FULL
CitcomSFull_LDADD = $(top_builddir)/lib/libCitcomS.la $(MPILIBS)
CitcomSRegional_SOURCES = Citcom.c
CitcomSRegional_CPPFLAGS =
CitcomSRegional_LDADD = $(top_builddir)/lib/libCitcomS.la $(MPILIBS)

# citcoms (top-level Python script)
do_subst = sed \
	-e 's|[@]INTERPRETER[@]|$(INTERPRETER)|g' \
	-e 's|[@]PYTHONPATH[@]|$(PYTHONPATH)|g'
citcoms: $(srcdir)/citcoms.in
	$(do_subst) < $(srcdir)/citcoms.in > $@ || (rm -f $@ && exit 1)
CLEANFILES = $(nodist_bin_SCRIPTS)
EXTRA_DIST = citcoms.in

# pycitcoms (libCitcomS + CitcomSmodule + libExchanger +
# Exchangermodule + embedded Python interpreter)
pycitcoms_SOURCES = pycitcoms.cc
pycitcoms_LDADD = \
	$(top_builddir)/module/libPyxMPImodule.a \
	$(top_builddir)/module/Exchanger/libExchangermodule.a \
	$(top_builddir)/module/libCitcomSmodule.a \
	$(top_builddir)/lib/libCitcomS.la
pycitcoms$(EXEEXT): $(pycitcoms_OBJECTS) $(pycitcoms_DEPENDENCIES) 
	@rm -f pycitcoms$(EXEEXT)
	$(CXXLINK) $(PYTHON_LDFLAGS) $(PYTHON_LINKFORSHARED) \
		$(pycitcoms_LDFLAGS) $(pycitcoms_OBJECTS) $(pycitcoms_LDADD) \
		-lExchanger -l_mpimodule -ljournal \
		$(PYTHON_BLDLIBRARY) \
		$(MPILIBS) \
		$(FCLIBS) \
		$(PYTHON_LIBS) $(PYTHON_MODLIBS) $(PYTHON_SYSLIBS) \
		$(LIBS) \
		$(PYTHON_LDLAST)

## end of Makefile.am
