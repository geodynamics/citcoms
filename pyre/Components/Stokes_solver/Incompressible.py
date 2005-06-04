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

        import pyre.inventory as prop


        Solver = prop.str("Solver", default="cgrad",
                 validator=prop.choice(["cgrad",
                                        "multigrid",
                                        "multigrid-el"]))
        node_assemble = prop.bool("node_assemble", default=True)
        precond = prop.bool("precond", default=True)

        accuracy = prop.float("accuracy", default=1.0e-6)
        tole_compressibility = prop.float("tole_compressibility", default=1.0e-7)
        mg_cycle = prop.int("mg_cycle", default=1)
        down_heavy = prop.int("down_heavy", default=3)
        up_heavy = prop.int("up_heavy", default=3)

        vlowstep = prop.int("vlowstep", default=1000)
        vhighstep = prop.int("vhighstep", default=3)
        piterations = prop.int("piterations", default=1000)


# version
__id__ = "$Id: Incompressible.py,v 1.16 2005/06/03 21:51:45 leif Exp $"

# End of file
