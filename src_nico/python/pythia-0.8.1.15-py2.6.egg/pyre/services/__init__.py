#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

def evaluator(name=None):
    from Evaluator import Evaluator
    return Evaluator(name)


def pickler(name=None):
    from Pickler import Pickler
    return Pickler(name)


def request(command, args=None):
    from ServiceRequest import ServiceRequest
    return ServiceRequest(command, args)


# version
__id__ = "$Id: __init__.py,v 1.1.1.1 2005/03/08 16:13:48 aivazis Exp $"

# End of file 
