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

PROJECT = examples
PACKAGE = shock

#--------------------------------------------------------------------------
#

all: tidy

shock:
	shock.py #--pulse.generator=bath

# version
# $Id: Make.mm,v 1.1.1.1 2005/03/08 16:14:00 aivazis Exp $

# End of file
