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
        import pyre.properties

        # component modules
        import CitcomS.Components.Sphere as Sphere
        import CitcomS.Components.Exchanger as Exchanger

        inventory = [

            Mesher("mesher", default=Sphere.regionalSphere("regional-sphere")),
            pyre.facilities.facility("exchanger", default=Exchanger.finegridexchanger("exchanger", "exchanger")),

            pyre.properties.str("datafile", default="regtest")

            ]



# version
__id__ = "$Id: RegionalSolver.py,v 1.33 2003/10/01 22:07:40 tan2 Exp $"

# End of file
