#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.components.Component import Component


class Stokes_solver(Component):


    def __init__(self, name, facility="vsolver"):
        Component.__init__(self, name, facility)
        return



    def setProperties(self):
        import CitcomS.Regional as Regional
        #Regional.Stokes_solver_set_prop(self.inventory)
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
        import CitcomS.Regional as Regional
        if self.inventory.Solver == "cgrad":
            Regional.set_cg_defaults()
        elif self.inventory.Solver == "multigrid":
            Regional.set_mg_defaults()
        elif self.inventory.Solver == "multigrid-el":
            Regional.set_mg_el_defaults()
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



    class Inventory(Component.Inventory):

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
__id__ = "$Id: Stokes_solver.py,v 1.8 2003/07/15 21:47:05 tan2 Exp $"

# End of file
