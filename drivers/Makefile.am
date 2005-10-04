## Process this file with automake to produce Makefile.in
# $Id$

CC = $(MPICC)
INCLUDES = -I$(top_srcdir)/lib/Common $(MPIINCLUDES)
bin_PROGRAMS = CitcomSFull CitcomSRegional
CitcomSFull_SOURCES = Citcom.c
CitcomSFull_LDADD = $(top_builddir)/lib/Full/libCitcomSFull.la $(MPILIBS)
CitcomSRegional_SOURCES = Citcom.c
CitcomSRegional_LDADD = $(top_builddir)/lib/Regional/libCitcomSRegional.la $(MPILIBS)

## end of Makefile.am
