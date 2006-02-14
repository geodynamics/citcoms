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

PROJECT = journal
PACKAGE = tests/libjournal

PROJ_TIDY += $(PROJ_CPPTESTS)

PROJ_TESTS = $(PROJ_CPPTESTS)
PROJ_CPPTESTS = control diagnostics debug error firewall info null warning state

PROJ_LIBRARIES = -ljournal

#------------------------------------------------------------------------

all: tidy

test: $(PROJ_TESTS) 
	for test in ${PROJ_TESTS}; do ./$${test}; done

#------------------------------------------------------------------------

diagnostics: diagnostics.cc $(BLD_LIBDIR)/libjournal.$(EXT_SAR)
	$(CXX) $(CXXFLAGS) -o $@ diagnostics.cc $(LCXXFLAGS) $(PROJ_LIBRARIES)

control: control.cc $(BLD_LIBDIR)/libjournal.$(EXT_SAR)
	$(CXX) $(CXXFLAGS) -o $@ control.cc $(LCXXFLAGS) $(PROJ_LIBRARIES)

debug: debug.cc $(BLD_LIBDIR)/libjournal.$(EXT_SAR)
	$(CXX) $(CXXFLAGS) -o $@ debug.cc $(LCXXFLAGS) $(PROJ_LIBRARIES)

error: error.cc $(BLD_LIBDIR)/libjournal.$(EXT_SAR)
	$(CXX) $(CXXFLAGS) -o $@ error.cc $(LCXXFLAGS) $(PROJ_LIBRARIES)

firewall: firewall.cc $(BLD_LIBDIR)/libjournal.$(EXT_SAR)
	$(CXX) $(CXXFLAGS) -o $@ firewall.cc $(LCXXFLAGS) $(PROJ_LIBRARIES)

info: info.cc $(BLD_LIBDIR)/libjournal.$(EXT_SAR)
	$(CXX) $(CXXFLAGS) -o $@ info.cc $(LCXXFLAGS) $(PROJ_LIBRARIES)

null: null.cc $(BLD_LIBDIR)/libjournal.$(EXT_SAR)
	$(CXX) $(CXXFLAGS) -o $@ null.cc $(LCXXFLAGS) $(PROJ_LIBRARIES)

warning: warning.cc $(BLD_LIBDIR)/libjournal.$(EXT_SAR)
	$(CXX) $(CXXFLAGS) -o $@ warning.cc $(LCXXFLAGS) $(PROJ_LIBRARIES)

state: state.cc $(BLD_LIBDIR)/libjournal.$(EXT_SAR)
	$(CXX) $(CXXFLAGS) -o $@ state.cc $(LCXXFLAGS) $(PROJ_LIBRARIES)

# version
# $Id: Make.mm,v 1.1.1.1 2005/03/08 16:13:54 aivazis Exp $

# End of file
