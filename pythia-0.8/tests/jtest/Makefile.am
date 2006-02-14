## Process this file with automake to produce Makefile.in

subpackage = jtest
include $(top_srcdir)/packages/subpackage.am

# jtest
nobase_pkgpyexec_PYTHON = \
	jtest/__init__.py

# jtestmodule
subpkgpyexec_LTLIBRARIES = jtestmodule.la
jtestmodule_la_SOURCES = \
	jtestmodule/bindings.cc \
	jtestmodule/bindings.h \
	jtestmodule/exceptions.cc \
	jtestmodule/exceptions.h \
	jtestmodule/jtestmodule.cc \
	jtestmodule/misc.cc \
	jtestmodule/misc.h
jtestmodule_la_LDFLAGS = -module
jtestmodule_la_LIBADD = $(top_builddir)/packages/journal/libjournal/libjournal.la

# tests
TESTS =
INSTALL_TESTS = \
	tests/cpp.py \
	tests/diagnostics.py \
	tests/signon.py
EXTRA_DIST = $(INSTALL_TESTS)
installcheck-local:
	$(MAKE) $(AM_MAKEFLAGS) TESTS="$(INSTALL_TESTS)" check

# version
# $Id: Makefile.am,v 1.2 2005/09/20 23:33:51 leif Exp $

## end of Makefile.am
