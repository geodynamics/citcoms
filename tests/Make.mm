# -*- Makefile -*-
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

PROJECT = CitcomS
PACKAGE = tests

PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/$(PACKAGE)

PROJ_PYTHONTESTS = signon.py
PROJ_CPPTESTS = array2d
PROJ_EMPTYTESTS = $(BLD_BINDIR)/CitcomSFull $(BLD_BINDIR)/CitcomSRegional
PROJ_TESTS = $(PROJ_PYTHONTESTS) $(PROJ_CPPTESTS) $(PROJ_EMPTYTESTS)

#--------------------------------------------------------------------------
#

all: $(PROJ_TESTS)

test:
	for test in $(PROJ_TESTS) ; do $${test}; done;

release: tidy
	cvs release .

update: clean
	cvs update .


#--------------------------------------------------------------------------
#

array2d: array2d.cc ../module/Exchanger/Array2D.h ../module/Exchanger/Array2D.cc
	$(CXX) $(CXXFLAGS) $(LCXXFLAGS) -o $@ array2d.cc -L/homegurnis/tools/mpich-1.2.5-absoft-3.0/lib -L$(TOOLS_DIR)/lib -ljournal -lmpich -lpmpich



# version
# $Id: Make.mm,v 1.3 2003/10/30 22:25:03 tan2 Exp $

# End of file
