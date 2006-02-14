## Process this file with automake to produce Makefile.in
# $Id: Makefile.am,v 1.1 2005/09/09 16:12:03 leif Exp $

subpackage = blade
include $(top_srcdir)/packages/subpackage.am

SUBDIRS = etc

# blade
nobase_pkgpython_PYTHON = \
	blade/components/Blade.py \
	blade/components/__init__.py \
	blade/components/Inspector.py \
	blade/__init__.py \
	blade/pml/CodecPML.py \
	blade/pml/__init__.py \
	blade/pml/Parser.py \
	blade/pml/Renderer.py

# tests
EXTRA_DIST = \
	tests/blade.pml \
	tests/sample.py

## end of Makefile.am
