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
##	if not self._been_here:
##	    self.CitcomModule.PG_timestep_init(self.all_variables)
##	    self._been_here = True

##        dt = self.stable_timestep()
        self.CitcomModule.PG_timestep_solve(self.all_variables, dt)
	return



    def stable_timestep(self):
        dt = self.CitcomModule.stable_timestep(self.all_variables)
        return dt



    class Inventory(CitcomComponent.Inventory):

        import pyre.properties as prop

        inventory = [

            prop.bool("ADV", True),
            prop.float("fixed_timestep", 0.0),
            prop.float("finetunedt", 0.7),

            prop.int("adv_sub_iterations", 2),
            prop.float("maxadvtime", 10.0),

            prop.bool("aug_lagr", True),
            prop.float("aug_number", 2.0e3),

	    ]


# version
__id__ = "$Id: Advection_diffusion.py,v 1.16 2003/08/28 22:37:39 ces74 Exp $"

# End of file
