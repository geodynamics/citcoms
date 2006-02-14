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
PACKAGE = inventory/validators


#--------------------------------------------------------------------------
#

all: export

#--------------------------------------------------------------------------
# export

EXPORT_PYTHON_MODULES = \
    And.py \
    Binary.py \
    Choice.py \
    Greater.py \
    GreaterEqual.py \
    Less.py \
    LessEqual.py \
    Not.py \
    Or.py \
    Range.py \
    Ternary.py \
    Unary.py \
    Validator.py \
    __init__.py \


export:: export-package-python-modules

# version
# $Id: Make.mm,v 1.2 2005/03/10 04:03:20 aivazis Exp $

# End of file
