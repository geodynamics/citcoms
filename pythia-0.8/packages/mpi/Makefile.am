## Process this file with automake to produce Makefile.in
# $Id: Makefile.am,v 1.2 2005/09/20 23:33:50 leif Exp $

subpackage = mpi
include $(top_srcdir)/packages/subpackage.am

SUBDIRS = mpipython tests

CXX = $(MPICXX)
CXXLD = @CXX@
INCLUDES += $(MPIINCLUDES)

# mpi
nobase_pkgpyexec_PYTHON = \
	mpi/Application.py \
	mpi/CartesianCommunicator.py \
	mpi/CommunicatorGroup.py \
	mpi/Communicator.py \
	mpi/DummyCommunicator.py \
	mpi/__init__.py \
	mpi/LauncherMPICH.py \
	mpi/LauncherPBS.py \
	mpi/Launcher.py \
	mpi/Port.py

# mpimodule
lib_LTLIBRARIES = lib_mpimodule.la
subpkgpyexec_LTLIBRARIES = _mpimodule.la
subpkginclude_HEADERS = \
	mpimodule/Communicator.h \
	mpimodule/Group.h
lib_mpimodule_la_SOURCES = \
	mpimodule/Communicator.cc \
	mpimodule/Group.cc
lib_mpimodule_la_LIBADD = $(top_builddir)/packages/journal/libjournal/libjournal.la
_mpimodule_la_SOURCES = \
	mpimodule/bindings.cc \
	mpimodule/bindings.h \
	mpimodule/communicators.cc \
	mpimodule/communicators.h \
	mpimodule/exceptions.cc \
	mpimodule/exceptions.h \
	mpimodule/groups.cc \
	mpimodule/groups.h \
	mpimodule/misc.cc \
	mpimodule/misc.h \
	mpimodule/_mpimodule.cc \
	mpimodule/ports.cc \
	mpimodule/ports.h \
	mpimodule/startup.cc \
	mpimodule/startup.h
_mpimodule_la_LDFLAGS = -module
_mpimodule_la_LIBADD = \
	lib_mpimodule.la \
	$(top_builddir)/packages/journal/libjournal/libjournal.la

# export
clean-local: clean-subpkgincludeHEADERS
BUILT_SOURCES = export-subpkgincludeHEADERS
CLEANFILES = export-subpkgincludeHEADERS

## end of Makefile.am
