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
bin_SCRIPTS =

abs_top_srcdir = @abs_top_srcdir@
abs_top_builddir = @abs_top_builddir@
abs_builddir = @abs_builddir@

if COND_PYRE

bin_SCRIPTS += citcoms coupledcitcoms

if COND_EMBEDDING
    bin_PROGRAMS += pycitcoms
    INTERPRETER = $(bindir)/pycitcoms
    noinstINTERPRETER = $(abs_builddir)/pycitcoms
    noinstPYTHONPATH = $(abs_top_srcdir)
    libCitcomS = $(top_builddir)/lib/libCitcomS.a
else
    INTERPRETER = $(PYTHON)
    noinstINTERPRETER = $(PYTHON)
    noinstPYTHONPATH = $(abs_top_srcdir):$(abs_top_builddir)/module/.libs:$(abs_top_builddir)/module/Exchanger/.libs
    libCitcomS = $(top_builddir)/lib/libCitcomS.la
endif

if COND_EXCHANGER
    MAYBE_EXCHANGER_CPPFLAGS = -DWITH_EXCHANGER
    MAYBE_EXCHANGER_INCLUDES = -I$(top_srcdir)/module/Exchanger
    MAYBE_EXCHANGER_LDADD = $(top_builddir)/module/Exchanger/libExchangerLibmodule.a
else
    MAYBE_EXCHANGER_CPPFLAGS =
    MAYBE_EXCHANGER_INCLUDES =
    MAYBE_EXCHANGER_LDADD =
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
	$(MAYBE_EXCHANGER_INCLUDES) \
	$(PYTHON_EGG_CPPFLAGS) \
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
do_build = sed -e s%[@]INTERPRETER[@]%$(noinstINTERPRETER)%g -e s%[@]PYTHONPATH[@]%$(noinstPYTHONPATH)%g
do_install = sed -e s%[@]INTERPRETER[@]%$(INTERPRETER)%g -e s%[@]PYTHONPATH[@]%$(PYTHONPATH)%g
citcoms: $(srcdir)/citcoms.in Makefile
	$(do_build) < $(srcdir)/citcoms.in > $@ || (rm -f $@ && exit 1)
	chmod +x $@
coupledcitcoms: $(srcdir)/coupledcitcoms.in Makefile
	$(do_build) < $(srcdir)/coupledcitcoms.in > $@ || (rm -f $@ && exit 1)
	chmod +x $@
install-binSCRIPTS: $(bin_SCRIPTS)
	@$(NORMAL_INSTALL)
	test -z "$(bindir)" || $(mkdir_p) "$(DESTDIR)$(bindir)"
	@list='$(bin_SCRIPTS)'; for p in $$list; do \
	  if test -f "$$p"; then d=; else d="$(srcdir)/"; fi; \
	  if test -f $$d$$p; then \
	    f=`echo "$$p" | sed 's|^.*/||;$(transform)'`; \
	    echo " $(do_install) '$$d$$p.in' > '$(DESTDIR)$(bindir)/$$f'"; \
	    $(do_install) "$$d$$p.in" > "$(DESTDIR)$(bindir)/$$f"; \
	    echo " chmod +x '$(DESTDIR)$(bindir)/$$f'"; \
	    chmod +x "$(DESTDIR)$(bindir)/$$f"; \
	  else :; fi; \
	done
EXTRA_DIST = citcoms.in

# pycitcoms (libCitcomS + CitcomSLibmodule + libExchanger +
# ExchangerLibmodule + embedded Python interpreter)
pycitcoms_CPPFLAGS = $(MAYBE_EXCHANGER_CPPFLAGS)
pycitcoms_SOURCES = pycitcoms.cc
pycitcoms_LDADD = \
	$(top_builddir)/module/libPyxMPImodule.a \
	$(MAYBE_EXCHANGER_LDADD) \
	$(top_builddir)/module/libCitcomSLibmodule.a \
	$(libCitcomS) \
	$(LIBHDF5) \
	$(LIBM)
pycitcoms$(EXEEXT): $(pycitcoms_OBJECTS) $(pycitcoms_DEPENDENCIES) 
	@rm -f pycitcoms$(EXEEXT)
	$(CXXLINK) $(PYTHON_LDFLAGS) $(PYTHON_LINKFORSHARED) \
		$(pycitcoms_LDFLAGS) $(pycitcoms_OBJECTS) $(pycitcoms_LDADD) \
		$(PYTHON_EGG_LDFLAGS) \
		$(PYTHON_BLDLIBRARY) \
		$(MPILIBS) \
		$(FCLIBS) \
		$(PYTHON_LIBS) $(PYTHON_MODLIBS) $(PYTHON_SYSLIBS) \
		$(LIBS) \
		$(PYTHON_LDLAST)

## end of Makefile.am
