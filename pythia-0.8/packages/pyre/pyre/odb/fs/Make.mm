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
PACKAGE = odb/fs


#--------------------------------------------------------------------------
#

all: export

#--------------------------------------------------------------------------
# export

EXPORT_PYTHON_MODULES = \
    CodecODB.py \
    Curator.py \
    Depository.py \
    FileLocking.py \
    FileLockingNT.py \
    FileLockingPosix.py \
    Shelf.py \
    Vault.py \
    __init__.py


export:: export-package-python-modules


# version
# $Id: Make.mm,v 1.1.1.1 2005/03/08 16:13:41 aivazis Exp $

# End of file
