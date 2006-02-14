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
PACKAGE = tests/journal

PROJ_TESTS = channels.py info.py journald.py remote.py renderer.py signon.py

#------------------------------------------------------------------------

all: tidy

test:
	for test in ${PROJ_TESTS}; do ./$${test}; done

# version
# $Id: Make.mm,v 1.1.1.1 2005/03/08 16:13:54 aivazis Exp $

# End of file
