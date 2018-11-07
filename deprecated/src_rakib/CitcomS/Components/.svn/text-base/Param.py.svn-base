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

class Param(CitcomComponent):


    def __init__(self, name="param", facility="param"):
        CitcomComponent.__init__(self, name, facility)
        return



    def setProperties(self, stream):
        from CitcomSLib import Param_set_properties
        Param_set_properties(self.all_variables, self.inventory, stream)
        return



    class Inventory(CitcomComponent.Inventory):


        import pyre.inventory


        reference_state = pyre.inventory.int("reference_state", default=1)
        refstate_file = pyre.inventory.str("refstate_file", default="refstate.dat")

        mineral_physics_model = pyre.inventory.int("mineral_physics_model", default=3)

        file_vbcs = pyre.inventory.bool("file_vbcs", default=False)
        vel_bound_file = pyre.inventory.str("vel_bound_file", default="bvel.dat")

        file_tbcs = pyre.inventory.bool("file_tbcs", default=False)
        temp_bound_file = pyre.inventory.str("temp_bound_file", default="btemp.dat")

        mat_control = pyre.inventory.bool("mat_control", default=False)
        mat_file = pyre.inventory.str("mat_file", default="mat.dat")

        lith_age = pyre.inventory.bool("lith_age", default=False)
        lith_age_file = pyre.inventory.str("lith_age_file", default="age.dat")
        lith_age_time = pyre.inventory.bool("lith_age_time", default=False)
        lith_age_depth = pyre.inventory.float("lith_age_depth", default=0.0314)

        #DESCRIBE = pyre.inventory.bool("DESCRIBE", default=False)
        #BEGINNER = pyre.inventory.bool("BEGINNER", default=False)
        #VERBOSE = pyre.inventory.bool("VERBOSE", default=False)

        start_age = pyre.inventory.float("start_age", default=40.0)
        reset_startage = pyre.inventory.bool("reset_startage", default=False)



# version
__id__ = "$Id$"

# End of file
