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

class BC(CitcomComponent):


    def __init__(self, name="bc", facility="bc"):
        CitcomComponent.__init__(self, name, facility)
        return



    def setProperties(self):
        self.CitcomModule.BC_set_properties(self.all_variables, self.inventory)
        return



    def updatePlateVelocity(self):
        self.CitcomModule.BC_update_plate_velocity(self.all_variables)
        return



    class Inventory(CitcomComponent.Inventory):

        import pyre.inventory



        side_sbcs = pyre.inventory.bool("side_sbcs", default=False)
        pseudo_free_surf = pyre.inventory.bool("pseudo_free_surf", default=False)

        topvbc = pyre.inventory.int("topvbc", default=0)
        topvbxval = pyre.inventory.float("topvbxval", default=0.0)
        topvbyval = pyre.inventory.float("topvbyval", default=0.0)

        botvbc = pyre.inventory.int("botvbc", default=0)
        botvbxval = pyre.inventory.float("botvbxval", default=0.0)
        botvbyval = pyre.inventory.float("botvbyval", default=0.0)

        toptbc = pyre.inventory.int("toptbc", default=True)
        toptbcval = pyre.inventory.float("toptbcval", default=0.0)

        bottbc = pyre.inventory.int("bottbc", default=True)
        bottbcval = pyre.inventory.float("bottbcval", default=1.0)


	    # these parameters are for 'lith_age',
	    # put them here temporalily
        temperature_bound_adj = pyre.inventory.bool("temperature_bound_adj", default=False)
        depth_bound_adj = pyre.inventory.float("depth_bound_adj", default=0.157)
        width_bound_adj = pyre.inventory.float("width_bound_adj", default=0.08727)


# version
__id__ = "$Id$"

# End of file
