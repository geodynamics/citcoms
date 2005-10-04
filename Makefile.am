## Process this file with automake to produce Makefile.in
# $Id$

ACLOCAL_AMFLAGS = -I ./m4

if COND_PYRE
    MAYBE_PYRE = etc examples module pyre tests visual
endif

SUBDIRS = $(MAYBE_PYTHIA) $(MAYBE_EXCHANGER) lib drivers $(MAYBE_PYRE)
DIST_SUBDIRS = \
	$(MAYBE_DIST_PYTHIA) $(MAYBE_DIST_EXCHANGER) \
	lib drivers \
	etc examples module pyre tests visual

EXTRA_DIST = \
	doc/CitComS.doxyconf

## end of Makefile.am
