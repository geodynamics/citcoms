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

PROJECT = pyre
PACKAGE = inventory/properties


#--------------------------------------------------------------------------
#

all: export

#--------------------------------------------------------------------------
# export

EXPORT_PYTHON_MODULES = \
    Array.py \
    Bool.py \
    Dimensional.py \
    File.py \
    Float.py \
    InputFile.py \
    Integer.py \
    List.py \
    OutputFile.py \
    Preformatted.py \
    Slice.py \
    String.py \
    __init__.py \


export:: export-package-python-modules

# version
# $Id: Make.mm,v 1.2 2005/03/24 02:05:13 aivazis Exp $

# End of file
