#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from CitcomS.Components.CitcomComponent import CitcomComponent


class Imcompressible(CitcomComponent):


    def __init__(self, name, facility, CitcomModule):
        # bind component method to facility method
        CitcomModule.vsolver_set_properties = CitcomModule.Imcompressible_set_properties

        CitcomComponent.__init__(self, name, facility, CitcomModule)
        return



    def run(self):
        self.CitcomModule.general_stokes_solver(self.all_variables)
	return



    def init(self, parent):
        if self.inventory.Solver == "cgrad":
            self.CitcomModule.set_cg_defaults(self.all_variables)
        elif self.inventory.Solver == "multigrid":
            self.CitcomModule.set_mg_defaults(self.all_variables)
        elif self.inventory.Solver == "multigrid-el":
            self.CitcomModule.set_mg_el_defaults(self.all_variables)
	return



    #def fini(self):
	#return



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
__id__ = "$Id: Incompressible.py,v 1.9 2003/08/19 21:24:35 tan2 Exp $"

# End of file
