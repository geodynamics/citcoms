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


class RegionalApp(Application):


    def run(self):

        regional = self.inventory.solver
        regional.run()

        return



    class Inventory(Application.Inventory):

        import pyre.facilities
        import Solver

        inventory = [

            pyre.facilities.facility("solver", default=Solver.regionalSolver()),

            ]



# version
__id__ = "$Id: RegionalApp.py,v 1.29 2003/08/27 22:24:06 tan2 Exp $"

# End of file
