# -*- Makefile -*-
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PROJECT = CitcomS
PACKAGE = Coupler

PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/$(PACKAGE)

#--------------------------------------------------------------------------
#

all: export

release: tidy
	cvs release .

update: clean
	cvs update .

#--------------------------------------------------------------------------
#
# export

EXPORT_PYTHON_MODULES = \
	__init__.py \
	Coupler.py

export:: export-package-python-modules

# version
# $Id: Make.mm,v 1.1 2003/08/30 00:37:59 tan2 Exp $

# End of file
