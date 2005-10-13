#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                              Michael A.G. Aivazis
#                       California Institute of Technology
#                       (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from Source import Source
from AssetContainer import AssetContainer


class Project(AssetContainer):


    def identify(self, inspector):
        return inspector.onProject(self)


    def __init__(self, name):
        AssetContainer.__init__(self, name)
        return


# version
__id__ = "$Id: Project.py,v 1.1.1.1 2005/03/08 16:13:59 aivazis Exp $"

# End of file
