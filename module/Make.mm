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
# $Id: Make.mm,v 1.2 2003/03/24 17:38:20 tan2 Exp $

include local.def

PROJECT = CitcomS
PACKAGE = module 

BLD_DIRS = \
	Full \
	Regional

RECURSE_DIRS = $(BLD_DIRS)


all:
	BLD_ACTION="all" $(MM) recurse

tidy::
	BLD_ACTION="tidy" $(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse

distclean::
	BLD_ACTION="distclean" $(MM) recurse

# version
# $Id: Make.mm,v 1.2 2003/03/24 17:38:20 tan2 Exp $

#
# End of file
