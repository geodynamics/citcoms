#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from CitcomComponent import CitcomComponent

class Visc(CitcomComponent):


    def __init__(self, name="visc", facility="visc"):
        CitcomComponent.__init__(self, name, facility)

        self.inventory.Viscosity = "system"
        self.inventory.rheol = 3
        self.inventory.visc_smooth_method = 3
        return



    def setProperties(self):
        self.CitcomModule.Visc_set_properties(self.all_variables, self.inventory)
        return



    class Inventory(CitcomComponent.Inventory):


        import pyre.properties


        inventory = [

            pyre.properties.bool("VISC_UPDATE", True),

            pyre.properties.int("num_mat", 4),
            pyre.properties.sequence("visc0", [1, 1, 1, 1]),

            pyre.properties.bool("TDEPV", False),
            pyre.properties.sequence("viscE", [1, 1, 1, 1]),
            pyre.properties.sequence("viscT", [1, 1, 1, 1]),

            pyre.properties.bool("SDEPV",False),
            pyre.properties.sequence("sdepv_expt", [1, 1, 1, 1]),
            pyre.properties.float("sdepv_misfit", 0.02),

            pyre.properties.bool("VMIN", False),
            pyre.properties.float("visc_min", 1.0e-3),

            pyre.properties.bool("VMAX", False),
            pyre.properties.float("visc_max", 1.0e3)

            ]

# version
__id__ = "$Id: Visc.py,v 1.8 2003/10/28 23:51:48 tan2 Exp $"

# End of file
