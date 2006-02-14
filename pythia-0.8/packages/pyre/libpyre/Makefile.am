## Process this file with automake to produce Makefile.in
# $Id: Makefile.am,v 1.1 2005/09/09 16:12:07 leif Exp $

subpackage = pyre
include $(top_srcdir)/packages/subpackage.am

subpkginclude_HEADERS = \
	include/shareable.h
nobase_subpkginclude_HEADERS = \
	algebra/operators.h \
	algebra/operators.icc \
	containers/Initializer.h \
	containers/Initializer.icc \
	containers/Tuple.h \
	containers/Tuple.icc \
	containers/Vector.h \
	containers/Vector.icc \
	geometry/CanonicalMesh.h \
	geometry/CanonicalMesh.icc \
	geometry/Point.h \
	geometry/Point.icc \
	memory/FixedAllocator.h \
	memory/FixedAllocator.icc \
	memory/PtrShareable.h \
	memory/PtrShareable.icc \
	memory/Recycler.h \
	memory/Recycler.icc \
	memory/Shareable.h \
	memory/Shareable.icc \
	picklers/debug/point.h \
	picklers/debug/tuple.h \
	picklers/debug/vector.h \
	picklers/tecplot/mesh.h
EXTRA_DIST = \
	picklers/mathematica/point.h

# export
clean-local: clean-subpkgincludeHEADERS clean-nobase_subpkgincludeHEADERS
BUILT_SOURCES = export-subpkgincludeHEADERS export-nobase_subpkgincludeHEADERS
CLEANFILES = export-subpkgincludeHEADERS export-nobase_subpkgincludeHEADERS

## end of Makefile.am
