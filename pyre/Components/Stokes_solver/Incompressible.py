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

        self.inventory.mg_cycle = 1
        self.inventory.down_heavy = 3
        self.inventory.up_heavy = 3

        self.inventory.vlowstep = 1000
        self.inventory.vhighstep = 3
        self.inventory.piterations = 1000
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

	    ]

# version
__id__ = "$Id: Incompressible.py,v 1.14 2003/10/29 18:40:01 tan2 Exp $"

# End of file
