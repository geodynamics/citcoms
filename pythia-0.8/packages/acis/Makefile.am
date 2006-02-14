## Process this file with automake to produce Makefile.in
# $Id: Makefile.am,v 1.2 2005/09/20 23:33:48 leif Exp $

subpackage = acis
include $(top_srcdir)/packages/subpackage.am

SUBDIRS = tests

# acis
nobase_pkgpyexec_PYTHON = \
	acis/Body.py \
	acis/Entity.py \
	acis/Face.py \
	acis/Faceter.py \
	acis/__init__.py \
	acis/Pickler.py

# acismodule
INCLUDES += \
	-I$(ACIS_INCDIR)/mesh \
	-I$(ACIS_INCDIR)/base \
	-I$(ACIS_INCDIR)/bool \
	-I$(ACIS_INCDIR)/cstr \
	-I$(ACIS_INCDIR)/fct \
	-I$(ACIS_INCDIR)/ga \
	-I$(ACIS_INCDIR)/intr \
	-I$(ACIS_INCDIR)/kern \
	-I$(ACIS_INCDIR)/mmgr \
	-I$(ACIS_INCDIR)/spline
subpkgpyexec_LTLIBRARIES = acismodule.la
acismodule_la_SOURCES = \
	acismodule/acismodule.cc \
	acismodule/attributes.cc \
	acismodule/attributes.h \
	acismodule/bindings.cc \
	acismodule/bindings.h \
	acismodule/debug.cc \
	acismodule/debug.h \
	acismodule/entities.cc \
	acismodule/entities.h \
	acismodule/exceptions.cc \
	acismodule/exceptions.h \
	acismodule/faceting.cc \
	acismodule/faceting.h \
	acismodule/intersections.cc \
	acismodule/intersections.h \
	acismodule/Mesher.cc \
	acismodule/Mesher.h \
	acismodule/meshing.h \
	acismodule/misc.cc \
	acismodule/misc.h \
	acismodule/operators.cc \
	acismodule/operators.h \
	acismodule/solids.cc \
	acismodule/solids.h \
	acismodule/support.cc \
	acismodule/support.h \
	acismodule/transformations.cc \
	acismodule/transformations.h \
	acismodule/util.cc \
	acismodule/util.h
acismodule_la_LDFLAGS = -module
acismodule_la_LIBADD = \
	$(top_builddir)/packages/journal/libjournal/libjournal.la \
	-L$(ACIS_LIBDIR) \
	-lboolean \
	-lct_husk \
	-lga_husk \
	-leuler \
	-lfaceter \
	-lconstrct \
	-lintersct \
	-lkernel \
	-llawutil \
	-lbaseutil
if COND_ACIS_HAS_MESHER
    acismodule_la_SOURCES += acismodule/meshing.cc
    acismodule_la_LIBADD += -lmeshhusk
endif
EXTRA_DIST = \
	acismodule/imports \
	acismodule/pch.cc

## end of Makefile.am
