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
        self.inventory.visc_smooth_method = 3
        return



    def setProperties(self):
        inv = self.inventory
        inv.visc0 = map(float, inv.visc0)
        inv.viscE = map(float, inv.viscE)
        inv.viscT = map(float, inv.viscT)
        inv.viscZ = map(float, inv.viscZ)
        inv.sdepv_expt = map(float, inv.sdepv_expt)

        self.CitcomModule.Visc_set_properties(self.all_variables, inv)
        return



    def updateMaterial(self):
        self.CitcomModule.Visc_update_material(self.all_variables)
        return



    class Inventory(CitcomComponent.Inventory):


        import pyre.inventory



        VISC_UPDATE = pyre.inventory.bool("VISC_UPDATE", default=True)

        num_mat = pyre.inventory.int("num_mat", default=4)
        visc0 = pyre.inventory.list("visc0", default=[1, 1, 1, 1])

        TDEPV = pyre.inventory.bool("TDEPV", default=False)
        rheol = pyre.inventory.int("rheol", default=3)
        viscE = pyre.inventory.list("viscE", default=[1, 1, 1, 1])
        viscT = pyre.inventory.list("viscT", default=[1, 1, 1, 1])
        viscZ = pyre.inventory.list("viscZ", default=[1, 1, 1, 1])

        SDEPV = pyre.inventory.bool("SDEPV", default=False)
        sdepv_expt = pyre.inventory.list("sdepv_expt", default=[1, 1, 1, 1])
        sdepv_misfit = pyre.inventory.float("sdepv_misfit", default=0.02)

        VMIN = pyre.inventory.bool("VMIN", default=False)
        visc_min = pyre.inventory.float("visc_min", default=1.0e-3)

        VMAX = pyre.inventory.bool("VMAX", default=False)
        visc_max = pyre.inventory.float("visc_max", default=1.0e3)


# version
__id__ = "$Id: Visc.py,v 1.13 2005/06/03 21:51:44 leif Exp $"

# End of file
