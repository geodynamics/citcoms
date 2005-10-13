# -*- Makefile -*-
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        (C) 1998-2005 All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PROJECT = pyre

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
# $Id: Make.mm,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $

# End of file
