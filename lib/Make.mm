# -*- Makefile -*-
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#=====================================================================
#
#                              CitcomS
#                 ---------------------------------
#
#                              Authors:
#           Louis Moresi, Shijie Zhong, Lijie Han, Eh Tan,
#           Clint Conrad, Michael Gurnis, and Eun-seo Choi
#          (c) California Institute of Technology 1994-2005
#
#        By downloading and/or installing this software you have
#       agreed to the CitcomS.py-LICENSE bundled with this software.
#             Free for non-commercial academic research ONLY.
#      This program is distributed WITHOUT ANY WARRANTY whatsoever.
#
#=====================================================================
#
#  Copyright June 2005, by the California Institute of Technology.
#  ALL RIGHTS RESERVED. United States Government Sponsorship Acknowledged.
# 
#  Any commercial use must be negotiated with the Office of Technology
#  Transfer at the California Institute of Technology. This software
#  may be subject to U.S. export control laws and regulations. By
#  accepting this software, the user agrees to comply with all
#  applicable U.S. export laws and regulations, including the
#  International Traffic and Arms Regulations, 22 C.F.R. 120-130 and
#  the Export Administration Regulations, 15 C.F.R. 730-744. User has
#  the responsibility to obtain export licenses, or other export
#  authority as may be required before exporting such information to
#  foreign countries or providing access to foreign nationals.  In no
#  event shall the California Institute of Technology be liable to any
#  party for direct, indirect, special, incidental or consequential
#  damages, including lost profits, arising out of the use of this
#  software and its documentation, even if the California Institute of
#  Technology has been advised of the possibility of such damage.
# 
#  The California Institute of Technology specifically disclaims any
#  warranties, including the implied warranties or merchantability and
#  fitness for a particular purpose. The software and documentation
#  provided hereunder is on an "as is" basis, and the California
#  Institute of Technology has no obligations to provide maintenance,
#  support, updates, enhancements or modifications.
#
#=====================================================================
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
# $Id: Make.mm,v 1.8 2005/06/10 02:23:14 leif Exp $

#
# End of file
