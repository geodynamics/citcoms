# -*- Makefile -*-
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        (C) 1998-2003 All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PROJECT = CitcomS
PACKAGE = tests

PROJ_PYTHONTESTS = signon.py
PROJ_EMPTYTESTS = $(BLD_BINDIR)/CitcomSFull $(BLD_BINDIR)/CitcomSRegional
PROJ_TESTS = $(PROJ_PYTHONTESTS) $(PROJ_EMPTYTESTS)

#--------------------------------------------------------------------------
#

all: $(PROJ_TESTS)

test:
	for test in $(PROJ_TESTS) ; do $${test}; done;

release: tidy
	cvs release .

update: clean
	cvs update .

# version
# $Id: Make.mm,v 1.1 2003/03/24 01:46:37 tan2 Exp $

# End of file
