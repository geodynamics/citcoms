## Process this file with automake to produce Makefile.in
# $Id: Makefile.am,v 1.1 2005/09/20 23:33:49 leif Exp $

subpackage = journal
include $(top_srcdir)/packages/subpackage.am

subpkgpyexec_LTLIBRARIES = _journalmodule.la
_journalmodule_la_SOURCES = \
	_journalmodule.cc \
	ProxyDevice.cc \
	ProxyDevice.h \
	ProxyDevice.icc \
	bindings.cc \
	bindings.h \
	exceptions.cc \
	exceptions.h \
	facility.cc \
	facility.h \
	journal.cc \
	journal.h \
	misc.cc \
	misc.h \
	state.cc \
	state.h
_journalmodule_la_LDFLAGS = -module
_journalmodule_la_LIBADD = ../libjournal/libjournal.la

## end of Makefile.am
