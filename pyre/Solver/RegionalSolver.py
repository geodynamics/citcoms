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
        import pyre.inventory

        # component modules
        import CitcomS.Components.Sphere as Sphere


        mesher = Mesher("mesher", default=Sphere.regionalSphere("regional-sphere"))

        datafile = pyre.inventory.str("datafile", default="regtest")
        datafile_old = pyre.inventory.str("datafile_old", default="regtest")




# version
__id__ = "$Id: RegionalSolver.py,v 1.35 2005/06/03 21:51:46 leif Exp $"

# End of file
