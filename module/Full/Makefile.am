## Process this file with automake to produce Makefile.in
# $Id: Makefile.am,v 1.1 2005/08/24 22:45:51 leif Exp $

# Fullmodule is identical to Regionalmodule, except that it is linked
# against libCitcomSFull instead of libCitcomSRegional.

pkgpyexec_LTLIBRARIES = Fullmodule.la
INCLUDES = -I$(top_srcdir)/module/Regional -I$(PYTHON_INCDIR)
Fullmodule_la_SOURCES = Fullmodule.cc
Fullmodule_la_LDFLAGS = -module -release $(VERSION)
Fullmodule_la_LIBADD = \
	$(top_builddir)/module/Regional/libModuleCommon.la \
	$(top_builddir)/lib/Full/libCitcomSFull.la \
	-L$(PYTHIA_LIBDIR) -l_mpimodule -ljournal

## end of Makefile.am
