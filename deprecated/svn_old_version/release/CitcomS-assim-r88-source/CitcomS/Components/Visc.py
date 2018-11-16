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
        return



    def setProperties(self, stream):

        from CitcomSLib import Visc_set_properties

        inv = self.inventory
        inv.visc0 = map(float, inv.visc0)
        inv.z_layer = map(float, inv.z_layer)
        inv.viscE = map(float, inv.viscE)
        inv.viscT = map(float, inv.viscT)
        inv.viscZ = map(float, inv.viscZ)
        inv.sdepv_expt = map(float, inv.sdepv_expt)
        inv.pdepv_a = map(float, inv.pdepv_a)
        inv.pdepv_b = map(float, inv.pdepv_b)
        inv.pdepv_y = map(float, inv.pdepv_y)
        inv.cdepv_ff = map(float, inv.cdepv_ff)

        Visc_set_properties(self.all_variables, inv, stream)

        return



    def updateMaterial(self):
        from CitcomSLib import Visc_update_material
        Visc_update_material(self.all_variables)
        return



    class Inventory(CitcomComponent.Inventory):


        import pyre.inventory


        Viscosity = pyre.inventory.str("Viscosity", default="system")
        visc_smooth_method = pyre.inventory.int("visc_smooth_method", default=3)
        VISC_UPDATE = pyre.inventory.bool("VISC_UPDATE", default=True)

        num_mat = pyre.inventory.int("num_mat", default=4)
        visc0 = pyre.inventory.list("visc0", default=[1, 1, 1, 1])

        z_layer = pyre.inventory.list("z_layer", default=[-999, -999, -999,-999])
        visc_layer_control = pyre.inventory.bool("visc_layer_control", default=False)
        visc_layer_file = pyre.inventory.str("visc_layer_file", default="visc.dat")

        TDEPV = pyre.inventory.bool("TDEPV", default=False)
        rheol = pyre.inventory.int("rheol", default=3)
        viscE = pyre.inventory.list("viscE", default=[1, 1, 1, 1])
        viscT = pyre.inventory.list("viscT", default=[1, 1, 1, 1])
        viscZ = pyre.inventory.list("viscZ", default=[1, 1, 1, 1])

        SDEPV = pyre.inventory.bool("SDEPV", default=False)
        sdepv_expt = pyre.inventory.list("sdepv_expt", default=[1, 1, 1, 1])
        sdepv_misfit = pyre.inventory.float("sdepv_misfit", default=0.001)

        PDEPV = pyre.inventory.bool("PDEPV", default=False)
        pdepv_eff = pyre.inventory.bool("pdepv_eff", default=True)
        pdepv_a= pyre.inventory.list("pdepv_a", default=[1e20, 1e20, 1e20, 1e20])
        pdepv_b= pyre.inventory.list("pdepv_b", default=[0, 0, 0, 0])
        pdepv_y= pyre.inventory.list("pdepv_y", default=[1e20, 1e20, 1e20, 1e20])
        pdepv_offset = pyre.inventory.float("pdepv_offset", default=0.0)

        CDEPV = pyre.inventory.bool("CDEPV", default=False)
        cdepv_ff= pyre.inventory.list("cdepv_ff", default=[1])

        low_visc_channel = pyre.inventory.bool("low_visc_channel", default=False)
        low_visc_wedge = pyre.inventory.bool("low_visc_wedge", default=False)

        lv_min_radius = pyre.inventory.float("lv_min_radius", default=0.9764)
        lv_max_radius = pyre.inventory.float("lv_max_radius", default=0.9921)
        lv_channel_thickness = pyre.inventory.float("lv_channel_thickness", default=0.0047)
        lv_reduction = pyre.inventory.float("lv_reduction", default=0.5)

        VMIN = pyre.inventory.bool("VMIN", default=False)
        visc_min = pyre.inventory.float("visc_min", default=1.0e-3)

        VMAX = pyre.inventory.bool("VMAX", default=False)
        visc_max = pyre.inventory.float("visc_max", default=1.0e3)


# version
__id__ = "$Id: Visc.py 15789 2009-10-08 18:01:00Z tan2 $"

# End of file
