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

PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/$(PACKAGE)

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
# $Id: Make.mm,v 1.2 2003/04/05 20:25:10 tan2 Exp $

# End of file
