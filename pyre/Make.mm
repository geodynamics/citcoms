# -*- Makefile -*-
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

include local.def

PROJECT = CitcomS
PACKAGE = pyre

PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/$(PACKAGE)

BLD_DIRS = \
	Components \
	Controller \
	Facilities \
	Solver

RECURSE_DIRS = $(BLD_DIRS)


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
	CoupledRegionalApp.py \
	FullApp.py \
	RegionalApp.py


export:: export-python-modules
	BLD_ACTION="export" $(MM) recurse


# version
# $Id: Make.mm,v 1.10 2003/08/28 23:10:26 ces74 Exp $

#
# End of file
