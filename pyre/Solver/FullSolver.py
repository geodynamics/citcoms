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
        import pyre.properties

        # component modules
        import CitcomS.Components.Sphere as Sphere
        import CitcomS.Components.Exchanger as Exchanger

        inventory = [

            Mesher("mesher", default=Sphere.fullSphere("full-sphere")),
            pyre.facilities.facility("exchanger", default=Exchanger.coarsegridexchanger("exchanger", "exchanger")),

            pyre.properties.str("datafile", default="fulltest")

            ]



# version
__id__ = "$Id: FullSolver.py,v 1.9 2003/10/01 22:07:40 tan2 Exp $"

# End of file
