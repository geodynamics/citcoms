#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2003  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.components.Component import Component

class Advection_diffusion(Component):

    def __init__(self,name,facility="tsolver"):
        Component.__init__(self, name, facility)
        self._setProperties()
        return

    def _setProperties(self):
        import CitcomS.Regional as Regional
        #Regional.general_stokes_solver_set_prop(self.inventory)
        return

    def run(self):
        #test
        print self.inventory.ADV
        return

    def _init(self):
	return


    def _fini(self):
	return

    def _solve(self):
	return


    def _output(self, *args, **kwds):
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
__id__ = "$Id: Advection_diffusion.py,v 1.5 2003/07/09 19:42:27 tan2 Exp $"

# End of file
