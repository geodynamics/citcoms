#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.components.Component import Component

class Visc(Component):


    def __init__(self):
        Component.__init__(self, "visc", "visc")
        return



    def setProperties(self):
        import CitcomS.Regional as Regional
	Regional.Visc_set_prop(self.inventory)
        return



    class Inventory(Component.Inventory):


        import pyre.properties


        inventory = [

            pyre.properties.str("Viscosity","system"),
            pyre.properties.int("rheol",3),
            pyre.properties.int("visc_smooth_method",3),
            pyre.properties.bool("VISC_UPDATE",True),

            pyre.properties.int("num_mat",4),
            pyre.properties.sequence("visc0",
				     [1.0e3,2.0e-3,2.0e0,2.0e1]),

            pyre.properties.bool("TDEPV",True),
            pyre.properties.sequence("viscE",
				     [0.1, 0.1, 1.0, 1.0]),
            pyre.properties.sequence("viscT",
				     [-1.02126,-1.01853, -1.32722, -1.32722]),

            pyre.properties.bool("SDEPV",False),
            pyre.properties.sequence("sdepv_expt",
				     [1,1,1,1]),
            pyre.properties.float("sdepv_misfit",0.02),

            pyre.properties.bool("VMIN",True),
            pyre.properties.float("visc_min",1.0e-4),

            pyre.properties.bool("VMAX",True),
            pyre.properties.float("visc_max",1.0e3)

            ]

# version
__id__ = "$Id: Visc.py,v 1.3 2003/07/13 22:58:12 tan2 Exp $"

# End of file
