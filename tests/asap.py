#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2003  All Rights Reserved
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

    from mpi.StagingMPICH import StagingMPICH
    stager = StagingMPICH()
    
    # initialize using the local default values
    props = stager.properties
    props.nodes = 4
    props.nodegen = "a%03d"
    props.nodelist = "[51-60]"

    # return the component
    return stager


# version
__id__ = "$Id: asap.py,v 1.1 2003/05/23 17:58:13 tan2 Exp $"

# End of file 
