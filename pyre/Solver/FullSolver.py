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


    def __init__(self, name="full", facility="citcom"):
	Citcom.__init__(self, name, facility)
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
        from CitcomS.Components.Param import Param
        from CitcomS.Components.Phase import Phase
        from CitcomS.Components.Visc import Visc

        inventory = [

            Mesher("mesher", Sphere.fullSphere("full-sphere")),

            ]



# version
__id__ = "$Id: FullSolver.py,v 1.5 2003/08/27 20:52:46 tan2 Exp $"

# End of file
