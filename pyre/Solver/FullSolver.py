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

        # component modules
        import CitcomS.Components.Sphere as Sphere

        inventory = [

            Mesher("mesher", Sphere.fullSphere("full-sphere")),

            ]



# version
__id__ = "$Id: FullSolver.py,v 1.6 2003/08/27 22:24:07 tan2 Exp $"

# End of file
