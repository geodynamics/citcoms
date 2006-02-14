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


from pyre.parsing.locators.Traceable import Traceable


class Asset(Traceable):


    def identify(self, inspector):
        return inspector.onAsset(self)


    def __init__(self, name):
        Traceable.__init__(self)
        
        self.name = name
        return
    

# version
__id__ = "$Id: Asset.py,v 1.1.1.1 2005/03/08 16:13:59 aivazis Exp $"

# End of file
