# -*- Makefile -*-
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

PROJECT = weaver
PACKAGE = mills

#--------------------------------------------------------------------------
#

all: export


#--------------------------------------------------------------------------
# export

EXPORT_ETC = \
    c.odb \
    cxx.odb \
    csh.odb \
    f77.odb \
    f90.odb \
    html.odb \
    make.odb \
    perl.odb \
    python.odb \
    sh.odb \
    tex.odb \
    xml.odb \
    __vault__.odb


export:: export-etc

# version
# $Id: Make.mm,v 1.1.1.1 2005/03/08 16:13:39 aivazis Exp $

# End of file
