## Process this file with automake to produce Makefile.in
# $Id: Makefile.am,v 1.2 2005/09/20 23:33:50 leif Exp $

subpackage = pulse
include $(top_srcdir)/packages/subpackage.am

CXX = $(MPICXX)
CXXLD = @CXX@
INCLUDES += $(MPIINCLUDES)

# etc
dist_subpkgsysconf_DATA = etc/__vault__.odb etc/bath.odb etc/heaviside.odb

# pulse
nobase_pkgpyexec_PYTHON = \
	pulse/Bath.py \
	pulse/HeavisidePulse.py \
	pulse/__init__.py \
	pulse/Load.py \
	pulse/PressurePulse.py

# pulsemodule
subpkgpyexec_LTLIBRARIES = pulsemodule.la
pulsemodule_la_SOURCES = \
	pulsemodule/bindings.cc \
	pulsemodule/bindings.h \
	pulsemodule/driver.cc \
	pulsemodule/driver.h \
	pulsemodule/exceptions.cc \
	pulsemodule/exceptions.h \
	pulsemodule/generators.cc \
	pulsemodule/generators.h \
	pulsemodule/misc.cc \
	pulsemodule/misc.h \
	pulsemodule/pulsemodule.cc
pulsemodule_la_LDFLAGS = -module
pulsemodule_la_LIBADD = $(top_builddir)/packages/journal/libjournal/libjournal.la

# tests
EXTRA_DIST = \
	tests/signon.py \
	tests/solver.py

## end of Makefile.am
