#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from mpi.Application import Application
import journal


class FullApp(Application):


    def run(self):

        full = self.inventory.solver
        full.run()

        return



    class Inventory(Application.Inventory):

        import pyre.facilities
        import Solver

        inventory = [

            pyre.facilities.facility("solver", default=Solver.fullSolver()),

            ]



# version
__id__ = "$Id: FullApp.py,v 1.5 2003/08/27 22:24:06 tan2 Exp $"

# End of file
