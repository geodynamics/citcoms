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
# $Id: Make.mm,v 1.1 2003/03/24 01:46:37 tan2 Exp $

# End of file
