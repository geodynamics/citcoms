## Process this file with automake to produce Makefile.in
# $Id: Makefile.am,v 1.2 2005/09/20 23:33:50 leif Exp $

subpackage = rigid
include $(top_srcdir)/packages/subpackage.am

CXX = $(MPICXX)
CXXLD = @CXX@
INCLUDES += $(MPIINCLUDES)

# rigid
nobase_pkgpyexec_PYTHON = \
	rigid/__init__.py \
	rigid/Rigid.py

# rigidmodule
subpkgpyexec_LTLIBRARIES = rigidmodule.la
rigidmodule_la_SOURCES = \
	rigidmodule/bindings.cc \
	rigidmodule/bindings.h \
	rigidmodule/driver.cc \
	rigidmodule/driver.h \
	rigidmodule/exceptions.cc \
	rigidmodule/exceptions.h \
	rigidmodule/misc.cc \
	rigidmodule/misc.h \
	rigidmodule/rigidmodule.cc
rigidmodule_la_LDFLAGS = -module
rigidmodule_la_LIBADD = $(top_builddir)/packages/journal/libjournal/libjournal.la

# tests
EXTRA_DIST = \
	tests/signon.py \
	tests/solver.py

## end of Makefile.am
