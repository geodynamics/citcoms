# -*- Makefile -*-
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PROJECT = elc
PACKAGE = elc

#--------------------------------------------------------------------------
#

all: export


#--------------------------------------------------------------------------
#
# export

EXPORT_PYTHON_MODULES = \
    Boundary.py \
    Exchanger.py \
    ICEExchanger.py \
    MPIExchanger.py \
    SerialExchanger.py \
    SynchronizedExchanger.py \
    __init__.py


export:: export-python-modules

# version
# $Id: Make.mm,v 1.1.1.1 2005/03/08 16:13:28 aivazis Exp $

# End of file
