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

        # component modules
        import CitcomS.Components.Sphere as Sphere

        inventory = [

            Mesher("mesher", default=Sphere.regionalSphere("regional-sphere")),

            ]



# version
__id__ = "$Id: RegionalSolver.py,v 1.31 2003/08/29 20:41:04 tan2 Exp $"

# End of file
