#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

import journal


# the required component factory
def staging():

    info = journal.debug("staging")
    info.log("instantiating application launcher")

    from mpi.LauncherMPICH import LauncherMPICH
    stager = LauncherMPICH()

    # initialize using the local values
    stager.inventory.nodegen = "n%03d"
    stager.inventory.nodes = 12
    stager.inventory.nodelist = [101,102,103,104,105,106]
    return stager


# version
# $Id: hrothgar.py,v 1.2 2003/08/01 22:57:42 tan2 Exp $

# End of file
