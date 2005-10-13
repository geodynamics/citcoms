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
PROJ_CLEAN += $(PROJ_CPPTESTS)

PROJ_PYTESTS = \
    signon.py \
    raw.py \
    basic.py \
    cartesian.py \
    groups.py \

PROJ_TESTS = $(PROJ_PYTESTS)


#--------------------------------------------------------------------------
#

all: $(PROJ_TESTS)

test:
	for test in $(PROJ_TESTS) ; do mpirun -np 4 `which mpipython.exe` $${test}; done

release: tidy
	cvs release .

update: clean
	cvs update .

# version
# $Id: Make.mm,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $

# End of file
