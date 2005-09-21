## Process this file with automake to produce Makefile.in
# $Id$

pkgpyexec_LTLIBRARIES = Regionalmodule.la
noinst_LTLIBRARIES = libModuleCommon.la
CXX = $(MPICXX)
CXXLD = @CXX@
INCLUDES = \
	-I$(top_srcdir)/lib/Common \
	-I$(PYTHON_INCDIR) \
	$(MPIINCLUDES)

# libModuleCommon
libModuleCommon_la_SOURCES = \
	advdiffu.cc \
	advdiffu.h \
	bindings.cc \
	bindings.h \
	exceptions.cc \
	exceptions.h \
	initial_conditions.cc \
	initial_conditions.h \
	mesher.cc \
	mesher.h \
	misc.cc \
	misc.h \
	outputs.cc \
	outputs.h \
	setProperties.cc \
	setProperties.h \
	stokes_solver.cc \
	stokes_solver.h

# Regionalmodule
Regionalmodule_la_SOURCES = Regionalmodule.cc
Regionalmodule_la_LDFLAGS = -module -release $(VERSION)
Regionalmodule_la_LIBADD = \
	libModuleCommon.la \
	$(top_builddir)/lib/Regional/libCitcomSRegional.la \
	-l_mpimodule -ljournal

## end of Makefile.am
