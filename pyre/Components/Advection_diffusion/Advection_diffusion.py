#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.components.Component import Component
import CitcomS.Regional as Regional

class Advection_diffusion(Component):


    def __init__(self,name,facility="tsolver"):
        Component.__init__(self, name, facility)
        return



    def setProperties(self):
        import CitcomS.Regional as Regional
        Regional.Advection_diffusion_set_properties(self.inventory)
        return



    def run(self):
        #test
        print "ADV = ",self.inventory.ADV
        return

        self._solve()
        return


    def init(self,parent):
        Regional.set_convection_defaults()
        Regional.PG_timestep_init()
	return


    def fini(self):
	return

    def _solve(self):
        Regional.PG_timestep_solve()
	return


    def output(self, *args, **kwds):
	return


    class Inventory(Component.Inventory):

        import pyre.properties as prop

        inventory = [

            prop.bool("ADV",True),
            prop.float("fixed_timestep",0.0),
            prop.float("finetunedt",0.7),

            prop.int("adv_sub_iterations",2),
            prop.float("maxadvtime",10.0),

            prop.bool("precond",True),
            prop.bool("aug_lagr",True),
            prop.float("aug_number",2.0e3),

	    ]


# version
__id__ = "$Id: Advection_diffusion.py,v 1.8 2003/07/24 00:04:04 tan2 Exp $"

# End of file
