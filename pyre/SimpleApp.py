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


class SimpleApp(Application):


    def __init__(self, name="citcom"):
        Application.__init__(self, name)

        self.solver = None
        self.solverCommunicator = None
        self._info = journal.debug("application")
        return



    def run(self):
        self.initialize()
        self.reportConfiguration()
        self.launch()
        return



    def initialize(self):
        layout = self.findLayout()

        controller = self.inventory.controller
        controller.initialize(self)
        return



    def launch(self):
        controller = self.inventory.controller
        controller.launch(self)

        controller.march(steps=self.inventory.steps)
        return



    def findLayout(self):
        self.solver = self.inventory.solver
        import mpi
        self.solverCommunicator = mpi.world()
        return



    def reportConfiguration(self):

        import mpi
        rank = mpi.world().rank

        if rank != 0:
            return

        self._info.line("configuration:")
        self._info.line("  properties:")
        self._info.line("    name: %r" % self.inventory.name)
        self._info.line("    full name: %r" % self.inventory.fullname)

        self._info.line("  facilities:")
        self._info.line("    journal: %r" % self.inventory.journal.name)
        self._info.line("    staging: %r" % self.inventory.staging.name)

        self._info.line("    solver: %r" % self.inventory.solver.name)
        self._info.line("    controller: %r" % self.inventory.controller.name)

        return



    class Inventory(Application.Inventory):

        import pyre.facilities
        from CitcomS.Facilities.Solver import Solver as SolverFacility

        import Controller
        import Solver

        inventory = [

            pyre.facilities.facility("controller", default=Controller.controller()),
            SolverFacility("solver", default=Solver.regionalSolver()),

            pyre.properties.int("steps", 1),

            ]



# version
__id__ = "$Id: SimpleApp.py,v 1.5 2003/09/03 21:18:56 tan2 Exp $"

# End of file
