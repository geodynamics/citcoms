#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from CitcomS.Components.CitcomComponent import CitcomComponent

class Stokes_solver(CitcomComponent):


    def __init__(self, name, facility, CitcomModule):
        # bind component method to facility method
        CitcomModule.vsolver_set_properties = CitcomModule.Stokes_solver_set_properties

        CitcomComponent.__init__(self, name, facility, CitcomModule)
        return


    def run(self):

	# test
	print "vlowstep = ", self.inventory.vlowstep
	return

	self._form_RHS()
	self._form_LHS()
	self._solve()

	return



    def init(self, parent):
        if self.inventory.Solver == "cgrad":
            self.CitcomModule.set_cg_defaults()
        elif self.inventory.Solver == "multigrid":
            self.CitcomModule.set_mg_defaults()
        elif self.inventory.Solver == "multigrid-el":
            self.CitcomModule.set_mg_el_defaults()
	return



    def fini(self):
	return


    def _form_RHS(self):
	return


    def _form_LHS(self):
	return


    def _solve(self):
	return


    def output(self, *args, **kwds):
	return



    class Inventory(CitcomComponent.Inventory):

        import pyre.properties as prop

        inventory = [
            prop.str("Solver","cgrad"),
            prop.bool("node_assemble",True),

            prop.int("mg_cycle",1),
            prop.int("down_heavy",1),
            prop.int("up_heavy",1),

            prop.int("vlowstep",2000),
            prop.int("vhighstep",3),
            prop.int("piterations",375),

            prop.float("accuracy",1.0e-6),
            prop.float("tole_compressibility",1.0e-7),

	    ]


# version
__id__ = "$Id: Stokes_solver.py,v 1.10 2003/07/24 17:46:47 tan2 Exp $"

# End of file
