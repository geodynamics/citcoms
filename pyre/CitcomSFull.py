#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from Citcom import Citcom
import Full as CitcomModule
import journal


class CitcomSFull(Citcom):


    def __init__(self, name="full"):
	Citcom.__init__(self, name)
	self.CitcomModule = CitcomModule
        return



    class Inventory(Citcom.Inventory):

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
            VSolver("vsolver", Stokes_solver.incompressibleNewtonian(CitcomModule)),
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
__id__ = "$Id: CitcomSFull.py,v 1.4 2003/08/25 19:34:40 tan2 Exp $"

# End of file
