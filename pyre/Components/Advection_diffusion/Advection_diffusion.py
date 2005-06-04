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

        import pyre.inventory as prop


        inputdiffusivity = prop.float("inputdiffusivity", default=1)
        fixed_timestep = prop.float("fixed_timestep", default=0.0)
        finetunedt = prop.float("finetunedt", default=0.9)
        filter_temp = prop.bool("filter_temp", default=True)



# version
__id__ = "$Id: Advection_diffusion.py,v 1.22 2005/06/03 21:51:44 leif Exp $"

# End of file
