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
PACKAGE = components


#--------------------------------------------------------------------------
#

all: export

#--------------------------------------------------------------------------
# export

EXPORT_PYTHON_MODULES = \
    Actor.py \
    AuthenticatingActor.py \
    GenericActor.py \
    Login.py \
    Logout.py \
    NYI.py \
    Registrar.py \
    Sentry.py \
    __init__.py


export:: export-package-python-modules

# version
# $Id: Make.mm,v 1.5 2005/05/02 18:10:18 pyre Exp $

# End of file
