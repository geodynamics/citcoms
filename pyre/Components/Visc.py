#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#
# CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
# Copyright (C) 2002-2005, California Institute of Technology.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
#</LicenseText>
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
__id__ = "$Id$"

# End of file
