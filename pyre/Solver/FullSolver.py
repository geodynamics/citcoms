#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from CitcomApp import CitcomApp
import CitcomS.Full as CitcomModule
import journal


class FullApp(CitcomApp):


    def __init__(self):
	CitcomApp.__init__(self, "full-citcoms")
	self.CitcomModule = CitcomModule
        return



    class Inventory(CitcomApp.Inventory):

        import pyre.facilities

        # facilities
        from CitcomS.Facilities.Mesher import Mesher
        from CitcomS.Facilities.TSolver import TSolver
	from CitcomS.Facilities.VSolver import VSolver

        # component modules
        import CitcomS.Components.Advection_diffusion as Advection_diffusion
        import CitcomS.Components.Sphere as Sphere
	import CitcomS.Components.Stokes_solver as Stokes_solver

        # components
        from CitcomS.Components.BC import BC
        from CitcomS.Components.Const import Const
        from CitcomS.Components.IC import IC
	from CitcomS.Components.Parallel import Parallel
	from CitcomS.Components.Param import Param
        from CitcomS.Components.Phase import Phase
        from CitcomS.Components.Visc import Visc

        inventory = [
            Mesher("mesher", Sphere.fullSphere(CitcomModule)),
            VSolver("vsolver", Stokes_solver.imcompressibleNewtonian(CitcomModule)),
            TSolver("tsolver", Advection_diffusion.temperature_diffadv(CitcomModule)),

            pyre.facilities.facility("bc",
				     default=BC("bc", "bc", CitcomModule)),
            pyre.facilities.facility("const",
				     default=Const("const", "const", CitcomModule)),
            pyre.facilities.facility("ic",
				     default=IC("ic", "ic", CitcomModule)),
	    pyre.facilities.facility("parallel",
				     default=Parallel("parallel", "parallel", CitcomModule)),
            pyre.facilities.facility("param",
				     default=Param("param", "param", CitcomModule)),
            pyre.facilities.facility("phase",
				     default=Phase("phase", "phase", CitcomModule)),
            pyre.facilities.facility("visc",
				     default=Visc("visc", "visc", CitcomModule)),

            ]



# version
__id__ = "$Id: FullSolver.py,v 1.1 2003/08/01 22:24:00 tan2 Exp $"

# End of file
