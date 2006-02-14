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
PACKAGE = services


#--------------------------------------------------------------------------
#

all: export

#--------------------------------------------------------------------------
# export

EXPORT_PYTHON_MODULES = \
    Evaluator.py \
    Marshaller.py \
    Pickler.py \
    RequestError.py \
    Service.py \
    ServiceRequest.py \
    Session.py \
    TCPService.py \
    TCPSession.py \
    UDPService.py \
    UDPSession.py \
    __init__.py


export:: export-package-python-modules

# version
# $Id: Make.mm,v 1.2 2005/03/14 22:59:59 aivazis Exp $

# End of file
