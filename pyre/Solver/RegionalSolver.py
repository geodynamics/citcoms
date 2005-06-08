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

        import pyre.inventory

        # component modules
        import CitcomS.Components.Sphere as Sphere


        mesher = pyre.inventory.facility("mesher", factory=Sphere.regionalSphere, args=("regional-sphere",))

        datafile = pyre.inventory.str("datafile", default="regtest")
        datafile_old = pyre.inventory.str("datafile_old", default="regtest")




# version
__id__ = "$Id: RegionalSolver.py,v 1.36 2005/06/08 01:55:33 leif Exp $"

# End of file
