#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from Solver import Solver
import journal


class FullSolver(Solver):


    def __init__(self, name, facility="solver"):
	Solver.__init__(self, name, facility)
        import CitcomS.Full as CitcomModule
	self.CitcomModule = CitcomModule
        return



    class Inventory(Solver.Inventory):

        # facilities
        from CitcomS.Facilities.Mesher import Mesher
        import pyre.facilities

        # component modules
        import CitcomS.Components.Sphere as Sphere
        import CitcomS.Components.Exchanger as Exchanger

        inventory = [

            Mesher("mesher", default=Sphere.fullSphere("full-sphere")),
            pyre.facilities.facility("exchanger", default=Exchanger.coarsegridexchanger("exchanger", "exchanger")),

            ]



# version
__id__ = "$Id: FullSolver.py,v 1.8 2003/09/05 19:49:15 tan2 Exp $"

# End of file
