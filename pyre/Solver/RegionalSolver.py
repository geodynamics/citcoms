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


class RegionalSolver(Solver):


    def __init__(self, name, facility="solver"):
	Solver.__init__(self, name, facility)
        import CitcomS.Regional as CitcomModule
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

            Mesher("mesher", default=Sphere.regionalSphere("regional-sphere")),
            pyre.facilities.facility("exchanger", default=Exchanger.finegridexchanger("exchanger", "exchanger")),

            ]



# version
__id__ = "$Id: RegionalSolver.py,v 1.32 2003/09/05 19:49:15 tan2 Exp $"

# End of file
