## Process this file with automake to produce Makefile.in
# $Id: Makefile.am,v 1.1 2005/09/20 23:33:49 leif Exp $

subpackage = journal
include $(top_srcdir)/packages/subpackage.am

DEFS = -DBLD_PROCEDURE # don't define HAVE_CONFIG_H; we have portinfo
subpkginclude_HEADERS = \
	debug.h \
	debuginfo.h \
	diagnostics.h \
	error.h \
	firewall.h \
	info.h \
	warning.h \
	macros.h \
	manipulators.h \
	manipulators.icc \
	manip-explicit.h \
	manip-explicit.icc \
	manip-templated.h \
	manip-templated.icc \
	Diagnostic.h \
	Diagnostic.icc \
	Index.h \
	Index.icc \
	NullDiagnostic.h \
	NullDiagnostic.icc \
	SeverityDebug.h \
	SeverityDebug.icc \
	SeverityError.h \
	SeverityError.icc \
	SeverityFirewall.h \
	SeverityFirewall.icc \
	SeverityInfo.h \
	SeverityInfo.icc \
	SeverityWarning.h \
	SeverityWarning.icc
lib_LTLIBRARIES = libjournal.la
libjournal_la_SOURCES = \
	debuginfo.cc \
	firewall.cc \
	Console.cc \
	Console.h \
	DefaultRenderer.cc \
	DefaultRenderer.h \
	DefaultRenderer.icc \
	Device.cc \
	Device.h \
	Device.icc \
	Diagnostic.cc \
	Entry.cc \
	Entry.h \
	Entry.icc \
	Facility.cc \
	Facility.h \
	Facility.icc \
	FacilityMap.cc \
	FacilityMap.h \
	FacilityMap.icc \
	Index.cc \
	Journal.cc \
	Journal.h \
	Journal.icc \
	Renderer.cc \
	Renderer.h \
	Renderer.icc \
	SeverityDebug.cc \
	SeverityError.cc \
	SeverityFirewall.cc \
	SeverityInfo.cc \
	SeverityWarning.cc \
	StreamDevice.cc \
	StreamDevice.h
libjournal_la_LDFLAGS =

# export
clean-local: clean-subpkgincludeHEADERS
BUILT_SOURCES = export-subpkgincludeHEADERS
CLEANFILES = export-subpkgincludeHEADERS

## end of Makefile.am
