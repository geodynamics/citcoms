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
# $Id: Make.mm,v 1.4 2003/04/05 20:25:09 tan2 Exp $

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
# $Id: Make.mm,v 1.4 2003/04/05 20:25:09 tan2 Exp $

#
# End of file
