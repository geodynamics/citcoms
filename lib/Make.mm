# -*- Makefile -*-
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#
# CitcomS by Louis Moresi, Shijie Zhong, Lijie Han, Eh Tan,
# Clint Conrad, Michael Gurnis, and Eun-seo Choi.
# Copyright (C) 1994-2005, California Institute of Technology.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
#</LicenseText>
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
	(cd Common; $(MM))

Full:
#	(cd Common; TYPE=Full $(MM))
	(cd Full; $(MM))

Regional:
#	(cd Common; TYPE=Regional $(MM))
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
# $Id$

#
# End of file
