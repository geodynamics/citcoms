#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.components.Component import Component

class Adv_solver(Component):


    def __init__(self):
        Component.__init__(self, "adv-solver", "adv-solver")
        return



    def setProperties(self):
        import CitcomS.Regional as Regional
	Regional.Adv_solver_set_prop(self.inventory)
        return



    class Inventory(Component.Inventory):

        import pyre.properties


        inventory = [

            pyre.properties.bool("ADV",True),
            pyre.properties.float("fixed_timestep",0.0),
            pyre.properties.float("finetunedt",0.7),

            pyre.properties.int("adv_sub_iterations",2),
            pyre.properties.float("maxadvtime",10.0),
            pyre.properties.bool("precond",True),

            pyre.properties.bool("aug_lagr",True),
            pyre.properties.float("aug_number",2.0e3)

            ]

# version
__id__ = "$Id: Adv_solver.py,v 1.2 2003/07/09 19:42:27 tan2 Exp $"

# End of file
