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

PROJECT = journal
PACKAGE = devices

#--------------------------------------------------------------------------
#

all: export

#--------------------------------------------------------------------------
#
# export

EXPORT_PYTHON_MODULES = \
    Console.py \
    Device.py \
    File.py \
    NetRenderer.py \
    Renderer.py \
    TCPDevice.py \
    TextFile.py \
    UDPDevice.py \
    __init__.py


export:: export-package-python-modules

# version
# $Id: Make.mm,v 1.2 2005/03/14 07:33:24 aivazis Exp $

# End of file
