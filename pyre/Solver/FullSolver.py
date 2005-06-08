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

        import pyre.inventory

        # component modules
        import CitcomS.Components.Sphere as Sphere


        mesher = pyre.inventory.facility("mesher", factory=Sphere.fullSphere, args=("full-sphere",))

        datafile = pyre.inventory.str("datafile", default="fulltest")
        datafile_old = pyre.inventory.str("datafile_old", default="fulltest")




# version
__id__ = "$Id: FullSolver.py,v 1.12 2005/06/08 01:55:33 leif Exp $"

# End of file
