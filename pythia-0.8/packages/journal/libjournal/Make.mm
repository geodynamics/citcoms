# -*- Makefile -*-
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

include local.def

PROJECT = journal
PACKAGE = libjournal

PROJ_SAR = $(BLD_LIBDIR)/$(PACKAGE).$(EXT_SAR)
PROJ_DLL = $(BLD_LIBDIR)/$(PACKAGE).$(EXT_SO)
PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/$(PACKAGE)
PROJ_CLEAN += $(PROJ_DLL) $(PROJ_SAR)

PROJ_SRCS = \
    debuginfo.cc \
    firewall.cc \
    Console.cc \
    DefaultRenderer.cc \
    Device.cc \
    Diagnostic.cc \
    Entry.cc \
    FacilityMap.cc \
    Index.cc \
    Journal.cc \
    Renderer.cc \
    SeverityDebug.cc \
    SeverityError.cc \
    SeverityFirewall.cc \
    SeverityInfo.cc \
    SeverityWarning.cc \
    StreamDevice.cc \


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

all: $(PROJ_SAR) export

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
export:: export-headers export-libraries export-binaries

EXPORT_HEADERS = \
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
    SeverityWarning.icc \


EXPORT_LIBS = $(PROJ_SAR)
EXPORT_BINS = $(PROJ_DLL)

# version
# $Id: Make.mm,v 1.4 2005/06/04 01:09:01 cummings Exp $

# End of file
