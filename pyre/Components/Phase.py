#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from CitcomComponent import CitcomComponent

class Phase(CitcomComponent):


    def __init__(self, name="phase", facility="phase"):
        CitcomComponent.__init__(self, name, facility)
        return



    def setProperties(self):
        self.CitcomModule.Phase_set_properties(self.all_variables, self.inventory)
        return


    class Inventory(CitcomComponent.Inventory):


        import pyre.inventory


        Ra_410 = pyre.inventory.float("Ra_410", default=0.0)
        clapeyron410 = pyre.inventory.float("clapeyron410", default=0.0235)
        transT410 = pyre.inventory.float("transT410", default=0.78)
        width410 = pyre.inventory.float("width410", default=0.0058)

        Ra_670 = pyre.inventory.float("Ra_670", default=0.0)
        clapeyron670 = pyre.inventory.float("clapeyron670", default=-0.0235)
        transT670 = pyre.inventory.float("transT670", default=0.78)
        width670 = pyre.inventory.float("width670", default=0.0058)

        Ra_cmb = pyre.inventory.float("Ra_cmb", default=0.0)
        clapeyroncmb = pyre.inventory.float("clapeyroncmb", default=-0.0235)
        transTcmb = pyre.inventory.float("transTcmb", default=0.875)
        widthcmb = pyre.inventory.float("widthcmb", default=0.0058)


# version
__id__ = "$Id: Phase.py,v 1.8 2005/06/03 21:51:44 leif Exp $"

# End of file
