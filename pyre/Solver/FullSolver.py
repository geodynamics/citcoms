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
        import pyre.properties

        # component modules
        import CitcomS.Components.Sphere as Sphere

        inventory = [

            Mesher("mesher", default=Sphere.fullSphere("full-sphere")),

            pyre.properties.str("datafile", default="fulltest"),
            pyre.properties.str("datafile_old", default="fulltest"),

            ]



# version
__id__ = "$Id: FullSolver.py,v 1.10 2003/10/28 23:51:49 tan2 Exp $"

# End of file
