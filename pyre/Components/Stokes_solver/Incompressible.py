#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from CitcomS.Components.CitcomComponent import CitcomComponent


class Incompressible(CitcomComponent):



    def run(self):
        self.CitcomModule.general_stokes_solver(self.all_variables)
	return



    def setup(self):
        if self.inventory.Solver == "cgrad":
            self.CitcomModule.set_cg_defaults(self.all_variables)
        elif self.inventory.Solver == "multigrid":
            self.CitcomModule.set_mg_defaults(self.all_variables)
        elif self.inventory.Solver == "multigrid-el":
            self.CitcomModule.set_mg_el_defaults(self.all_variables)
	return



    #def fini(self):
	#return



    def setProperties(self):
        self.CitcomModule.Incompressible_set_properties(self.all_variables, self.inventory)
        return



    class Inventory(CitcomComponent.Inventory):

        import pyre.properties as prop

        inventory = [
            prop.str("Solver", "cgrad"),
            prop.bool("node_assemble", True),
            prop.bool("precond",True),

            prop.int("mg_cycle", 1),
            prop.int("down_heavy", 3),
            prop.int("up_heavy", 3),

            prop.int("vlowstep", 500),
            prop.int("vhighstep", 3),
            prop.int("piterations", 500),

            prop.float("accuracy", 1.0e-6),
            prop.float("tole_compressibility", 1.0e-7),

	    ]

# version
__id__ = "$Id: Incompressible.py,v 1.12 2003/08/27 20:52:47 tan2 Exp $"

# End of file
