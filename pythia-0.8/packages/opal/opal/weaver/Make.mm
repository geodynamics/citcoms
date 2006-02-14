# -*- Makefile -*-
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PROJECT = opal
PACKAGE = weaver


#--------------------------------------------------------------------------
#

all: export

#--------------------------------------------------------------------------
# export

EXPORT_PYTHON_MODULES = \
    BodyMill.py \
    ContentMill.py \
    DocumentMill.py \
    HeadMill.py \
    PageMill.py \
    StructuralMill.py \
    TagMill.py \
    __init__.py


export:: export-package-python-modules

# version
# $Id: Make.mm,v 1.2 2005/03/27 11:05:04 aivazis Exp $

# End of file
