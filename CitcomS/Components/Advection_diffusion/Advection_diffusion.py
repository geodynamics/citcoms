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

from CitcomS.Components.CitcomComponent import CitcomComponent


class Advection_diffusion(CitcomComponent):


    def __init__(self, name, facility):
        CitcomComponent.__init__(self, name, facility)
        return



    def setProperties(self):
        from CitcomSLib import Advection_diffusion_set_properties
        Advection_diffusion_set_properties(self.all_variables, self.inventory)
        return



    def run(self,dt):
        self._solve(dt)
        return



    def setup(self):
        from CitcomSLib import set_convection_defaults
        set_convection_defaults(self.all_variables)
	self._been_here = False
	return


    def launch(self):
        from CitcomSLib import PG_timestep_init
        PG_timestep_init(self.all_variables)
        return

    #def fini(self):
	#return



    def _solve(self,dt):
        from CitcomSLib import PG_timestep_solve
        PG_timestep_solve(self.all_variables, dt)
	return



    def stable_timestep(self):
        from CitcomSLib import stable_timestep
        dt = stable_timestep(self.all_variables)
        return dt



    class Inventory(CitcomComponent.Inventory):

        import pyre.inventory as prop

        ADV = prop.bool("ADV", default=True)
        filter_temp = prop.bool("filter_temp", default=True)

        fixed_timestep = prop.float("fixed_timestep", default=0.0)
        finetunedt = prop.float("finetunedt", default=0.9)
        inputdiffusivity = prop.float("inputdiffusivity", default=1)

        adv_sub_iterations = prop.int("adv_sub_iterations", default=2)

        aug_lagr = prop.bool("aug_lagr", default=True)
        aug_number = prop.float("aug_number", default=2.0e3)



# version
__id__ = "$Id$"

# End of file
