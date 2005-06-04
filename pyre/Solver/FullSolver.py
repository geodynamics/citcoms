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
        import pyre.inventory

        # component modules
        import CitcomS.Components.Sphere as Sphere


        mesher = Mesher("mesher", default=Sphere.fullSphere("full-sphere"))

        datafile = pyre.inventory.str("datafile", default="fulltest")
        datafile_old = pyre.inventory.str("datafile_old", default="fulltest")




# version
__id__ = "$Id: FullSolver.py,v 1.11 2005/06/03 21:51:46 leif Exp $"

# End of file
