# -*- Makefile -*-
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                                  Steve Quenette
#                        California Institute of Technology
#                        (C) 1998-2003  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# version
# $Id: Make.mm,v 1.4 2003/04/25 22:58:28 tan2 Exp $

include local.def

PROJECT = CitcomS
PACKAGE = lib

PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/$(PACKAGE)

BLD_DIRS = \
	Common \
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
# $Id: Make.mm,v 1.4 2003/04/25 22:58:28 tan2 Exp $

#
# End of file
