# -*- Makefile -*-
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        (C) 1998-2002 All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PROJECT = CitcomS
PACKAGE = examples

PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/$(PACKAGE)

#--------------------------------------------------------------------------
#

all: clean

release: clean
	cvs release .

update: clean
	cvs update .

#--------------------------------------------------------------------------
#

clean::
	@find * -name \*.bak -exec rm {} \;
	@find * -name \*.pyc -exec rm {} \;
	@find * -name \*~ -exec rm {} \;

# version
# $Id: Make.mm,v 1.2 2003/04/05 20:25:09 tan2 Exp $

# End of file
