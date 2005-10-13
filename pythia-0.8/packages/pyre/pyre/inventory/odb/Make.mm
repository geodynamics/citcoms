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
PACKAGE = inventory/odb

EXPORT_ETCDIR = ${EXPORT_ROOT}/etc


#--------------------------------------------------------------------------
#

all: export

prefix:
	sed -e 's|xxDBROOTxx|${EXPORT_ROOT}|g' < prefix-template.py > prefix.py

#--------------------------------------------------------------------------
# export

EXPORT_PYTHON_MODULES = \
    Curator.py \
    Descriptor.py \
    Inventory.py \
    Registry.py \
    prefix.py \
    __init__.py


export:: prefix export-package-python-modules
	$(RM) $(RMFLAGS) prefix.py prefix-template.pyc


# version
# $Id: Make.mm,v 1.1.1.1 2005/03/08 16:13:43 aivazis Exp $

# End of file
