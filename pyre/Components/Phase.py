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


        import pyre.properties

        inventory = [

            pyre.properties.float("Ra_410", 0.0),
            pyre.properties.float("clapeyron410", 0.0235),
            pyre.properties.float("transT410", 0.78),
            pyre.properties.float("width410", 0.0058),

            pyre.properties.float("Ra_670", 0.0),
            pyre.properties.float("clapeyron670", -0.0235),
            pyre.properties.float("transT670", 0.78),
            pyre.properties.float("width670", 0.0058),

            pyre.properties.float("Ra_cmb", 0.0),
            pyre.properties.float("clapeyroncmb", -0.0235),
            pyre.properties.float("transTcmb", 0.875),
            pyre.properties.float("widthcmb", 0.0058)

            ]

# version
__id__ = "$Id: Phase.py,v 1.7 2003/10/28 23:51:48 tan2 Exp $"

# End of file
