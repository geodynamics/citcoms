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


    def __init__(self, name, facility, CitcomModule):
        # bind component method to facility method
        CitcomModule.tsolver_set_properties = CitcomModule.Advection_diffusion_set_properties

        CitcomComponent.__init__(self, name, facility, CitcomModule)
        return



    def run(self):
        self._solve()
        return



    def init(self,parent):
        self.CitcomModule.set_convection_defaults()
	self._been_here = False
	return



    #def fini(self):
	#return



    def _solve(self):
	if not self._been_here:
	    self.CitcomModule.PG_timestep_init()
	    self._been_here = True

        self.CitcomModule.PG_timestep_solve()
	return



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
__id__ = "$Id: Advection_diffusion.py,v 1.11 2003/07/28 21:57:02 tan2 Exp $"

# End of file
