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


from opal.applications.WebApplication import WebApplication as Base


class WebApplication(Base):


    def _getPrivateDepositoryLocations(self):
        return ['../content', '../config']


# version
__id__ = "$Id: WebApplication.py,v 1.2 2005/03/27 01:19:54 aivazis Exp $"

# End of file 
