# -*- Makefile -*-
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PROJECT = CitcomS
PACKAGE = Components

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
	Adv_solver.py \
	BC.py \
	Const.py \
	IC.py \
	Mesh.py \
	Parallel.py \
	Param.py \
	Phase.py \
	Visc.py


export:: export-package-python-modules

# version
# $Id: Make.mm,v 1.2 2003/06/27 00:15:02 tan2 Exp $

# End of file
