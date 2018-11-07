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


class Login(GenericActor):


    def __init__(self):
        GenericActor.__init__(self, 'login')
        return


# version
__id__ = "$Id: Login.py,v 1.3 2005/05/02 18:09:31 pyre Exp $"

# End of file 
