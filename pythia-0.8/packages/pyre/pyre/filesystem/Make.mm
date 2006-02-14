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
PACKAGE = filesystem


#--------------------------------------------------------------------------
#

all: export

#--------------------------------------------------------------------------
# export

EXPORT_PYTHON_MODULES = \
    BlockDevice.py \
    CharacterDevice.py \
    Directory.py \
    Entry.py \
    Explorer.py \
    File.py \
    FileSystem.py \
    Finder.py \
    Inspector.py \
    Link.py \
    NamedPipe.py \
    Root.py \
    SimpleRenderer.py \
    Socket.py \
    TreeRenderer.py \
    fastfind.py \
    __init__.py


export:: export-package-python-modules

# version
# $Id: Make.mm,v 1.1.1.1 2005/03/08 16:13:46 aivazis Exp $

# End of file
