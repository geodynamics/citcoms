# -*- Makefile -*-
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

PROJECT = CitcomS
PACKAGE = Exchanger

include std-pythonmodule.def


PROJ_SRCS = \
	Exchanger.cc \
	FineGridExchanger.cc \
	bindings.cc \
	exceptions.cc \
	misc.cc


# version
# $Id: Make.mm,v 1.1 2003/09/06 23:44:22 tan2 Exp $

# End of file
