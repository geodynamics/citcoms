# -*- Makefile -*-
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PROJECT = opaldemo
PACKAGE = bin

#--------------------------------------------------------------------------
#

all: tidy


daemons: tidy
	@echo "Launching journald (the journal daemon)"
	./journald.py
	sleep 1
	@echo "Launching ipad (the authentication daemon)"
	./ipad.py
	sleep 1
	@echo "Launching idd (the unique identifier daemon)"
	./idd.py


# version
# $Id: Make.mm,v 1.3 2005/04/28 03:51:13 pyre Exp $

# End of file
