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
PACKAGE = applications


#--------------------------------------------------------------------------
#

all: export

#--------------------------------------------------------------------------
# export

EXPORT_PYTHON_MODULES = \
    Application.py \
    ClientServer.py \
    CommandlineParser.py \
    ComponentHarness.py \
    Daemon.py \
    DynamicComponentHarness.py \
    Executive.py \
    Script.py \
    ServiceDaemon.py \
    ServiceHarness.py \
    Stager.py \
    __init__.py


export:: export-package-python-modules

# version
# $Id: Make.mm,v 1.2 2005/03/11 07:08:40 aivazis Exp $

# End of file
