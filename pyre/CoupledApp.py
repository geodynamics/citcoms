#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from SimpleApp import SimpleApp
import journal


class CoupledApp(SimpleApp):


    def __init__(self, name="citcom"):
        SimpleApp.__init__(self, name)

        self.solver = None
        self.solverCommunicator = None
        self.intercomm = None

        self.rank = 0
        self.nodes = 0
        self.leader = 0
        self.remoteLeader = 0

        self._info = journal.debug("application")
        return



    def initialize(self):
        layout = self.inventory.layout
        layout.initialize(self)

        self.findLayout(layout)

        controller = self.inventory.controller
        controller.initialize(self)

        return



    def findLayout(self, layout):
        import Components.Exchanger as Exchanger

        if layout.coarse:
            self.solver = self.inventory.coarse
            self.exchanger = self.inventory.cge
            self.solverCommunicator = layout.coarse
        elif layout.fine:
            self.solver = self.inventory.fine
            self.exchanger = self.inventory.fge
            self.solverCommunicator = layout.fine
        else:
            import journal
            journal.warning(self.name).log("node '%d' is an orphan" % layout.rank)

        self.intercomm = layout.intercomm
        self.rank = layout.rank
        self.nodes = layout.nodes
        self.leader = layout.leader
        self.remoteLeader = layout.remoteLeader

        return



    def reportConfiguration(self):

        import mpi
        rank = mpi.world().rank

        if rank != 0:
            return

        self._info.line("configuration:")
#        self._info.line("  properties:")
#        self._info.line("    name: %r" % self.inventory.name)
#        self._info.line("    full name: %r" % self.inventory.fullname)

        self._info.line("  facilities:")
        self._info.line("    journal: %r" % self.inventory.journal.name)
        self._info.line("    staging: %r" % self.inventory.staging.name)

        self._info.line("    coarse: %r" % self.inventory.coarse.name)
        self._info.line("    fine: %r" % self.inventory.fine.name)
        self._info.line("    cge: %r" % self.inventory.cge.name)
        self._info.line("    fge: %r" % self.inventory.fge.name)
        self._info.line("    controller: %r" % self.inventory.controller.name)
        self._info.line("    coupler: %r" % self.inventory.coupler.name)
        self._info.line("    layout: %r" % self.inventory.layout.name)

        return



    class Inventory(SimpleApp.Inventory):

        import pyre.facilities
        from CitcomS.Facilities.Solver import Solver as SolverFacility

        import Controller
        import Solver
        import Coupler
        import Layout
        import CitcomS.Components.Exchanger as Exchanger

        inventory = [

            pyre.facilities.facility("controller", default=Controller.controller()),
            pyre.facilities.facility("coupler", default=Coupler.coupler()),
            pyre.facilities.facility("layout", default=Layout.layout()),

            SolverFacility("coarse", default=Solver.fullSolver("coarse", "coarse")),
            SolverFacility("fine", default=Solver.regionalSolver("fine", "fine")),
            pyre.facilities.facility("cge", default=Exchanger.coarsegridexchanger()),
            pyre.facilities.facility("fge", default=Exchanger.finegridexchanger()),

            pyre.properties.int("steps", 1),

            ]



# version
__id__ = "$Id: CoupledApp.py,v 1.8 2003/10/29 01:13:15 tan2 Exp $"

# End of file
