#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.components.Component import Component

class Phase(Component):


    def __init__(self):
        Component.__init__(self, "phase", "phase")
        return



    def setProperties(self):
        import CitcomS.Regional as Regional
	Regional.Phase_set_prop(self.inventory)
        return



    class Inventory(Component.Inventory):


        import pyre.properties

        inventory = [

            pyre.properties.float("Ra_410",0.0),
            pyre.properties.float("clapeyron410",0.0235),
            pyre.properties.float("transT410",0.78),
            pyre.properties.float("width410",0.0058),

            pyre.properties.float("Ra_670",0.0),
            pyre.properties.float("clapeyron670",0.0235),
            pyre.properties.float("transT670",0.78),
            pyre.properties.float("width670",0.0058),

            pyre.properties.float("Ra_cmb",0.0),
            pyre.properties.float("clapeyroncmb",0.0235),
            pyre.properties.float("transTcmb",0.78),
            pyre.properties.float("widthcmb",0.0058)

            ]

# version
__id__ = "$Id: Phase.py,v 1.2 2003/07/09 19:42:27 tan2 Exp $"

# End of file
