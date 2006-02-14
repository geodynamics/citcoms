## Process this file with automake to produce Makefile.in
# $Id: Makefile.am,v 1.2 2005/09/20 23:33:48 leif Exp $

subpackage = elc
include $(top_srcdir)/packages/subpackage.am

SUBDIRS = tests
CXX = $(MPICXX)
CXXLD = @CXX@
INCLUDES += $(MPIINCLUDES)

# elc
nobase_pkgpyexec_PYTHON = \
	elc/Boundary.py \
	elc/Exchanger.py \
	elc/ICEExchanger.py \
	elc/__init__.py \
	elc/MPIExchanger.py \
	elc/SerialExchanger.py \
	elc/SynchronizedExchanger.py

# elcmodule
subpkgpyexec_LTLIBRARIES = elcmodule.la
elcmodule_la_SOURCES = \
	elcmodule/bindings.cc \
	elcmodule/bindings.h \
	elcmodule/elcmodule.cc \
	elcmodule/exceptions.cc \
	elcmodule/exceptions.h \
	elcmodule/memory.cc \
	elcmodule/memory.h \
	elcmodule/misc.cc \
	elcmodule/misc.h \
	elcmodule/verify.cc \
	elcmodule/verify.h \
	elcmodule/via_mpi.cc \
	elcmodule/via_mpi.h
elcmodule_la_LDFLAGS = -module
elcmodule_la_LIBADD = $(top_builddir)/packages/journal/libjournal/libjournal.la

# etc
dist_subpkgsysconf_DATA = etc/__vault__.odb etc/ice.odb etc/mpi.odb etc/serial.odb

## end of Makefile.am
