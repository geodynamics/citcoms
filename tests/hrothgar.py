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
def staging(options):

    info = journal.debug("staging")
    info.log("instantiating application launcher")

    from mpi.StagingMPICH import StagingMPICH
    stager = StagingMPICH()
    
    # initialize using the local default values
    stager.properties.nodegen = "n%03d"

    # initialize using the local default values
    stager.configure(options)

    return stager


# version
# $Id: hrothgar.py,v 1.1 2003/05/23 17:58:13 tan2 Exp $

# End of file 
