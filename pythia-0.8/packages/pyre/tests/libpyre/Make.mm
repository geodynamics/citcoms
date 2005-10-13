# -*- Makefile -*-
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005 All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PROJECT = pyre
PACKAGE = tests/libpyre

PROJ_CPPTESTS = \
    manip1 manip2 manip3 point recycler shareable tuple vector

PROJ_TESTS = $(PROJ_CPPTESTS)
PROJ_INCDIR = $(BLD_INCDIR)/$(PROJECT)/$(PACKAGE)
PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/$(PACKAGE)

PROJ_CLEAN += $(PROJ_CPPTESTS)
PROJ_LIBRARIES = 

#--------------------------------------------------------------------------
#

all: tidy

test:
	for test in $(PROJ_TESTS) ; do $${test}; done

release: tidy
	cvs release .

update: clean
	cvs update .

#--------------------------------------------------------------------------
#

PROJ_CXX_INCLUDES = ../..

manip1: manip1.cc
	$(CXX) $(CXXFLAGS) $(LCXXFLAGS) -o $@ manip1.cc

manip2: manip2.cc
	$(CXX) $(CXXFLAGS) $(LCXXFLAGS) -o $@ manip2.cc

manip3: manip3.cc
	$(CXX) $(CXXFLAGS) $(LCXXFLAGS) -o $@ manip3.cc

point: point.cc
	$(CXX) $(CXXFLAGS) $(LCXXFLAGS) -o $@ point.cc -ljournal

recycler: recycler.cc
	$(CXX) $(CXXFLAGS) $(LCXXFLAGS) -o $@ recycler.cc -ljournal

shareable: shareable.cc
	$(CXX) $(CXXFLAGS) $(LCXXFLAGS) -o $@ shareable.cc -ljournal

tuple: tuple.cc
	$(CXX) $(CXXFLAGS) $(LCXXFLAGS) -o $@ tuple.cc -ljournal

vector: vector.cc
	$(CXX) $(CXXFLAGS) $(LCXXFLAGS) -o $@ vector.cc -ljournal


# version
# $Id: Make.mm,v 1.1.1.1 2005/03/08 16:13:50 aivazis Exp $

# End of file
