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


    def run(self, *args, **kwds):

	# test
	print self.properties.Solver
	return


	self.init()
	self.form_RHS()
	self.form_LHS()
	self.solve()
	self.fini()

	return


    def init(self):
	return


    def fini(self):
	return


    def form_RHS(self):
	return


    def form_LHS(self):
	return


    def solve(self):
	return


    def output(self, *args, **kwds):
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


	    # the following propoerties should belong to 'Viscosity'
	    # put them here temporalily
            prop.string("Viscosity","system"),
            prop.int("rheol",3),
            prop.int("visc_smooth_method",3),
            prop.bool("VISC_UPDATE",True),
            prop.int("num_mat",4),

            prop.bool("TDEPV",True),
            prop.sequence("viscE",
                                     [0.1, 0.1, 1.0, 1.0]),
            prop.sequence("viscT",
                                     [-1.02126,-1.01853, -1.32722, -1.32722]),
            prop.sequence("visc0",
                                     [1.0e3,2.0e-3,2.0e0,2.0e1]),

            prop.bool("SDEPV",False),
            prop.sequence("sdepv_expt",
                                     [1,1,1,1]),
            prop.float("sdepv_misfit",0.02),

            prop.bool("VMIN",True),
            prop.float("visc_min",1.0e-4),

            prop.bool("VMAX",True),
            prop.float("visc_max",1.0e3),

	    )


# version
__id__ = "$Id: Stokes_solver.py,v 1.4 2003/06/23 20:54:13 tan2 Exp $"

# End of file
