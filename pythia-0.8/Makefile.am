## Process this file with automake to produce Makefile.in
# $Id: Makefile.am,v 1.2 2005/09/20 23:33:48 leif Exp $

include $(top_srcdir)/common.am

ACLOCAL_AMFLAGS = -I ./m4

SUBDIRS = packages tests
DIST_SUBDIRS = $(SUBDIRS) examples

pyrepythondir = $(pkgpyexecdir)/pyre
nodist_pyrepython_PYTHON = config.py
config.py: $(top_builddir)/portinfo $(top_builddir)/Makefile
	$(PYCONFIG) -h $(top_builddir)/portinfo -m $(top_builddir)/Makefile -o $@
CLEANFILES = config.py

## end of Makefile.am
