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


def actor(name):
    from GenericActor import GenericActor
    return GenericActor(name)


def authenticatingActor(name):
    from AuthenticatingActor import AuthenticatingActor
    return AuthenticatingActor(name)


def sentry(*args):
    from Sentry import Sentry
    return Sentry(*args)


# version
__id__ = "$Id: __init__.py,v 1.3 2005/03/27 20:42:44 aivazis Exp $"

# End of file 
