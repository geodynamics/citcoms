# -*- Makefile -*-
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005 All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PROJECT = pyre
PACKAGE = weaver/mills

#--------------------------------------------------------------------------
#

all: export

#--------------------------------------------------------------------------
# export

EXPORT_PYTHON_MODULES = \
    CMill.py \
    CshMill.py \
    CxxMill.py \
    Fortran77Mill.py \
    Fortran90Mill.py \
    HTMLMill.py \
    MakeMill.py \
    PerlMill.py \
    PythonMill.py \
    ShMill.py \
    TeXMill.py \
    XMLMill.py \
    __init__.py \


export:: export-package-python-modules

# version
# $Id: Make.mm,v 1.1.1.1 2005/03/08 16:13:47 aivazis Exp $

# End of file
