## Process this file with automake to produce Makefile.in
# $Id: Makefile.am,v 1.1 2005/08/24 22:45:51 leif Exp $

ACLOCAL_AMFLAGS = -I ./m4

if COND_PYRE
    MAYBE_PYRE = etc examples module pyre tests visual
endif
SUBDIRS = lib drivers $(MAYBE_PYRE)

EXTRA_DIST = \
	CitcomS.py-LICENSE.txt \
	doc/CitComS.doxyconf

## end of Makefile.am
