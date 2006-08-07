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
noinst_SCRIPTS =

if COND_PYRE

noinst_SCRIPTS += citcoms

if COND_EMBEDDING
    bin_PROGRAMS += pycitcoms
    INTERPRETER = $(bindir)/pycitcoms
    noinstINTERPRETER = $$here/pycitcoms
    noinstPYTHONPATH = $$top_srcdir
    libCitcomS = $(top_builddir)/lib/libCitcomS.a
else
    INTERPRETER = $(PYTHON)
    noinstINTERPRETER = $(PYTHON)
    noinstPYTHONPATH = $$top_srcdir:$$top_srcdir/module/.libs:$$top_srcdir/module/Exchanger/.libs
    libCitcomS = $(top_builddir)/lib/libCitcomS.la
endif

endif

CC = $(MPICC)
CXX = $(MPICXX)
INCLUDES = \
	-I$(top_srcdir)/lib \
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
CitcomSFull_LDADD = $(libCitcomS) $(MPILIBS) $(LIBHDF5) $(LIBM)
CitcomSRegional_SOURCES = Citcom.c
CitcomSRegional_CPPFLAGS =
CitcomSRegional_LDADD = $(libCitcomS) $(MPILIBS) $(LIBHDF5) $(LIBM)

if COND_HDF5
    CitcomSFull_CPPFLAGS += -DUSE_HDF5
    CitcomSRegional_CPPFLAGS += -DUSE_HDF5
endif

# citcoms (top-level Python script)
citcoms: $(srcdir)/citcoms.in Makefile
	here=`pwd`; \
	top_srcdir=`cd $(top_srcdir) && pwd`; \
	sed \
		-e "s|[@]INTERPRETER[@]|$(noinstINTERPRETER)|g" \
		-e "s|[@]PYTHONPATH[@]|$(noinstPYTHONPATH)|g" \
		< $(srcdir)/citcoms.in > $@ || (rm -f $@ && exit 1)
	chmod +x $@
install-exec-hook:
	here=`pwd`; \
	sed \
		-e "s|[@]INTERPRETER[@]|$(INTERPRETER)|g" \
		-e "s|[@]PYTHONPATH[@]|$(PYTHONPATH)|g" \
		< $(srcdir)/citcoms.in > $(DESTDIR)$(bindir)/citcoms || (rm -f $@ && exit 1)
	chmod +x $(DESTDIR)$(bindir)/citcoms
CLEANFILES = $(noinst_SCRIPTS)
EXTRA_DIST = citcoms.in

# pycitcoms (libCitcomS + CitcomSLibmodule + libExchanger +
# ExchangerLibmodule + embedded Python interpreter)
pycitcoms_SOURCES = pycitcoms.cc
pycitcoms_LDADD = \
	$(top_builddir)/module/libPyxMPImodule.a \
	$(top_builddir)/module/Exchanger/libExchangerLibmodule.a \
	$(top_builddir)/module/libCitcomSLibmodule.a \
	$(libCitcomS) \
	$(LIBHDF5) \
	$(LIBM)
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
