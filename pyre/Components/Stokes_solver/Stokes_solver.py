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
        self._setProperties()
        return



    def _setProperties(self):
        import CitcomS.Regional as Regional
        #Regional.general_stokes_solver_set_prop(self.properties)
        return



    def run(self):

	# test
	print self.properties.Solver
	return


	self._init()
	self._form_RHS()
	self._form_LHS()
	self._solve()
	self._fini()

	return



    def _init(self):
	return


    def _fini(self):
	return


    def _form_RHS(self):
	return


    def _form_LHS(self):
	return


    def _solve(self):
	return


    def _output(self, *args, **kwds):
	return



    class Properties(Component.Properties):

        import pyre.properties as prop

        __properties__ = (
            prop.string("Solver","cgrad"),
            prop.bool("node_assemble",True),

            prop.int("mg_cycle",1),
            prop.int("down_heavy",1),
            prop.int("up_heavy",1),

            prop.int("vlowstep",2000),
            prop.int("vhighstep",3),
            prop.int("piterations",375),

            prop.float("accuracy",1.0e-6),
            prop.float("tole_compressibility",1.0e-7),


	    )


# version
__id__ = "$Id: Stokes_solver.py,v 1.5 2003/06/26 23:14:08 tan2 Exp $"

# End of file
