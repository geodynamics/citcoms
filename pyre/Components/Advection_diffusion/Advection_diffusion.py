#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from CitcomS.Components.CitcomComponent import CitcomComponent


class Advection_diffusion(CitcomComponent):


    def __init__(self, name, facility):
        CitcomComponent.__init__(self, name, facility)
        self.inventory.ADV = True
        self.inventory.fixed_timestep = 0.0
        self.inventory.finetunedt = 0.7

        self.inventory.adv_sub_iterations = 2
        self.inventory.maxadvtime = 10

        self.inventory.aug_lagr = True
        self.inventory.aug_number = 2.0e3
        return



    def setProperties(self):
        self.CitcomModule.Advection_diffusion_set_properties(self.all_variables, self.inventory)
        return



    def run(self,dt):
        self._solve(dt)
        return



    def setup(self):
        self.CitcomModule.set_convection_defaults(self.all_variables)
	self._been_here = False
	return


    def launch(self):
        self.CitcomModule.PG_timestep_init(self.all_variables)
        return

    #def fini(self):
	#return



    def _solve(self,dt):
        self.CitcomModule.PG_timestep_solve(self.all_variables, dt)
	return



    def stable_timestep(self):
        dt = self.CitcomModule.stable_timestep(self.all_variables)
        return dt



    class Inventory(CitcomComponent.Inventory):

        import pyre.properties as prop

        inventory = [

            prop.float("inputdiffusivity", default=1),

	    ]


# version
__id__ = "$Id: Advection_diffusion.py,v 1.19 2003/11/28 22:17:26 tan2 Exp $"

# End of file
