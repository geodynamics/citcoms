#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from CitcomComponent import CitcomComponent

class Carrier(CitcomComponent):


    class Inventory(CitcomComponent.Inventory):

        import pyre.properties as prop


        inventory = [

            ]



# version
__id__ = "$Id: Exchanger.py,v 1.1 2003/08/30 00:39:17 tan2 Exp $"

# End of file
