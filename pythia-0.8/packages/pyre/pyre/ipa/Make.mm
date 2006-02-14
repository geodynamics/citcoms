# -*- Makefile -*-
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005 All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PROJECT = pyre
PACKAGE = ipa

#--------------------------------------------------------------------------
#

all: export

#--------------------------------------------------------------------------
#
# export

EXPORT_PYTHON_MODULES = \
    Authentication.py \
    Daemon.py \
    IPAService.py \
    IPASession.py \
    Pickler.py \
    UserManager.py \
    __init__.py


export:: export-package-python-modules

# version
# $Id: Make.mm,v 1.1.1.1 2005/03/08 16:13:40 aivazis Exp $

# End of file
