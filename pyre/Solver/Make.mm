# -*- Makefile -*-
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PROJECT = CitcomS
PACKAGE = Solver

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
	FullSolver.py \
	RegionalSolver.py \
	Solver.py \


export:: export-package-python-modules

# version
# $Id: Make.mm,v 1.1 2003/08/27 22:24:07 tan2 Exp $

# End of file
