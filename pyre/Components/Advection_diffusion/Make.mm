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
PACKAGE = Advection_diffusion

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
	Advection_diffusion.py \
	Temperature_diffadv.py


export:: export-package-python-modules

# version
# $Id: Make.mm,v 1.2 2003/07/03 23:43:20 ces74 Exp $

# End of file
