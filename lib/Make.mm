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
PACKAGE = lib

PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/$(PACKAGE)

BLD_DIRS = \
	Common \
	Full \
	Regional

.PHONY: Common
.PHONY: Full
.PHONY: Regional

all: $(BLD_DIRS)

# This make file has to compile the common directory twice, to create two
# distinct set of binaries... one with customisations for full/spherical
# runs and one for region runs. Hence a few hacks are needed.

Common:

Full:
	(cd Common; TYPE=Full $(MM))
	(cd Full; $(MM))

Regional:
	(cd Common; TYPE=Regional $(MM))
	(cd Regional; $(MM))

clean::
	(cd Common; TYPE=Full $(MM) clean)
	(cd Common; TYPE=Regional $(MM) clean)
	(cd Full;  $(MM) clean)
	(cd Regional; $(MM) clean)

distclean::
	(cd Common; TYPE=Full $(MM) distclean)
	(cd Common; TYPE=Regional $(MM) distclean)
	(cd Full; $(MM) distclean)
	(cd Regional; $(MM) distclean)

# version
# $Id: Make.mm,v 1.6 2003/08/07 18:41:41 tan2 Exp $

#
# End of file
