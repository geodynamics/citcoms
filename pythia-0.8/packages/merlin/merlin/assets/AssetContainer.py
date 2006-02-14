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

from Asset import Asset


class AssetContainer(Asset):


    def append(self, assets):
        self.assets += assets
        return


    def identify(self, inspector):
        return inspector.onAssetContainer(self)


    def __init__(self, name):
        Asset.__init__(self, name)
        self.assets = []
        return
    

# version
__id__ = "$Id: AssetContainer.py,v 1.1.1.1 2005/03/08 16:13:59 aivazis Exp $"

# End of file
