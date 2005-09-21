## Process this file with automake to produce Makefile.in
# $Id$

pkgpyexec_LTLIBRARIES = Exchangermodule.la
CXX = $(MPICXX)
CXXLD = @CXX@
INCLUDES = \
	-I$(top_srcdir)/lib/Common \
	-I$(PYTHON_INCDIR) \
	$(MPIINCLUDES)
Exchangermodule_la_SOURCES = \
	AreaWeightedNormal.cc \
	AreaWeightedNormal.h \
	bindings.cc \
	bindings.h \
	Boundary.cc \
	Boundary.h \
	CitcomInterpolator.cc \
	CitcomInterpolator.h \
	CitcomSource.cc \
	CitcomSource.h \
	Convertor.cc \
	Convertor.h \
	exceptions.cc \
	exceptions.h \
	Exchangermodule.cc \
	exchangers.cc \
	exchangers.h \
	global_bbox.cc \
	global_bbox.h \
	initTemperature.cc \
	initTemperature.h \
	inlets_outlets.cc \
	inlets_outlets.h \
	Interior.cc \
	Interior.h \
	misc.cc \
	misc.h \
	SInlet.cc \
	SInlet.h \
	SIUnit.cc \
	SIUnit.h \
	SVTInlet.cc \
	SVTInlet.h \
	SVTOutlet.cc \
	SVTOutlet.h \
	TInlet.cc \
	TInlet.h \
	TOutlet.cc \
	TOutlet.h \
	VOutlet.cc \
	VOutlet.h \
	VTInlet.cc \
	VTInlet.h \
	VTOutlet.cc \
	VTOutlet.h
EXTRA_DIST = \
	BoundaryVTInlet.cc \
	BoundaryVTInlet.h
Exchangermodule_la_LDFLAGS = -module -release $(VERSION)
Exchangermodule_la_LIBADD = \
	$(top_builddir)/lib/Regional/libCitcomSRegional.la \
	-lExchanger \
	-l_mpimodule -ljournal

## end of Makefile.am
