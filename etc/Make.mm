# -*- Makefile -*-
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

PROJECT = citcoms
PACKAGE =

#--------------------------------------------------------------------------
#

all: export


#--------------------------------------------------------------------------
# export

EXPORT_ETC = \
    __vault__.odb \
    full.odb \
    full-sphere.odb \
    incomp-newtonian.odb \
    incomp-non-newtonian.odb \
    regional.odb \
    regional-sphere.odb \
    temp.odb


export:: export-etc

# version
# $Id: Make.mm,v 1.1 2005/06/03 21:51:39 leif Exp $

# End of file
