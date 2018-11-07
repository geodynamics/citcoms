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


from GenericActor import GenericActor


class Logout(GenericActor):


    def __init__(self):
        GenericActor.__init__(self, 'logout')
        return


# version
__id__ = "$Id: Logout.py,v 1.2 2005/05/02 18:09:46 pyre Exp $"

# End of file 
