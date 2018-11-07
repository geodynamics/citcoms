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


def session(name=None):
    from IPASession import IPASession
    return IPASession(name)


def service(name=None):
    from IPAService import IPAService
    return IPAService(name)


def userManager(name=None):
    from UserManager import UserManager
    return UserManager(name)


def daemon(name=None):
    from Daemon import Daemon
    return Daemon(name)


def pickler(name=None):
    if name is None:
        name = 'ipa-pickler'

    from Pickler import Pickler
    return Pickler(name)


# version
__id__ = "$Id: __init__.py,v 1.2 2005/03/09 07:02:42 aivazis Exp $"

# End of file 
