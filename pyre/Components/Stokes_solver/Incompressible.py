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


    def __init__(self, name, facility):
        CitcomComponent.__init__(self, name, facility)

        return



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



    def launch(self):
        self.CitcomModule.general_stokes_solver_setup(self.all_variables)
        return



    #def fini(self):
	#return



    def setProperties(self):
        self.CitcomModule.Incompressible_set_properties(self.all_variables, self.inventory)
        return



    class Inventory(CitcomComponent.Inventory):

        import pyre.properties as prop

        inventory = [

            prop.str("Solver", "cgrad",
                     validator=prop.choice(["cgrad",
                                            "multigrid",
                                            "multigrid-el"])),
            prop.bool("node_assemble", True),
            prop.bool("precond", True),

            prop.float("accuracy", 1.0e-6),
            prop.float("tole_compressibility", 1.0e-7),
            prop.int("mg_cycle", 1),
            prop.int("down_heavy", 3),
            prop.int("up_heavy", 3),

            prop.int("vlowstep", 1000),
            prop.int("vhighstep", 3),
            prop.int("piterations", 1000),

	    ]

# version
__id__ = "$Id: Incompressible.py,v 1.15 2004/08/02 16:34:26 ces74 Exp $"

# End of file
