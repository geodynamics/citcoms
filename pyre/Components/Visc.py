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
        inv = self.inventory
        inv.visc0 = map(float, inv.visc0)
        inv.viscE = map(float, inv.viscE)
        inv.viscT = map(float, inv.viscT)
        inv.sdepv_expt = map(float, inv.sdepv_expt)

        self.CitcomModule.Visc_set_properties(self.all_variables, inv)
        return



    def updateMaterial(self):
        self.CitcomModule.Visc_update_material(self.all_variables)
        return



    class Inventory(CitcomComponent.Inventory):


        import pyre.properties


        inventory = [

            pyre.properties.bool("VISC_UPDATE", True),

            pyre.properties.int("num_mat", 4),
            pyre.properties.list("visc0", [1, 1, 1, 1]),

            pyre.properties.bool("TDEPV", False),
            pyre.properties.list("viscE", [1, 1, 1, 1]),
            pyre.properties.list("viscT", [1, 1, 1, 1]),

            pyre.properties.bool("SDEPV",False),
            pyre.properties.list("sdepv_expt", [1, 1, 1, 1]),
            pyre.properties.float("sdepv_misfit", 0.02),

            pyre.properties.bool("VMIN", False),
            pyre.properties.float("visc_min", 1.0e-3),

            pyre.properties.bool("VMAX", False),
            pyre.properties.float("visc_max", 1.0e3)

            ]

# version
__id__ = "$Id: Visc.py,v 1.10 2004/05/26 23:55:38 tan2 Exp $"

# End of file
