#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#
# CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
# Copyright (C) 2002-2005, California Institute of Technology.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
#</LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from SimpleApp import SimpleApp
import journal


class CoupledApp(SimpleApp):


    def __init__(self, name="CitcomS"):
        SimpleApp.__init__(self, name)

        self.solver = None
        self.solverCommunicator = None
        self.myPlus = []
        self.remotePlus = []

        self.comm = None
        self.rank = 0
        self.nodes = 0

        self._info = journal.debug("application")
        return



    def initialize(self):
        layout = self.inventory.layout
        layout.initialize(self)

        self.findLayout(layout)

        self.controller.initialize(self)

        return



    def findLayout(self, layout):

        if layout.coarse:
            self.controller = self.inventory.coarseController
            self.solver = self.inventory.coarse
            self.exchanger = self.inventory.cge
            self.solverCommunicator = layout.coarse
            self.myPlus = layout.coarsePlus
            self.remotePlus = layout.finePlus
        elif layout.fine:
            self.controller = self.inventory.fineController
            self.solver = self.inventory.fine
            self.exchanger = self.inventory.fge
            self.solverCommunicator = layout.fine
            self.myPlus = layout.finePlus
            self.remotePlus = layout.coarsePlus
        else:
            import journal
            journal.warning(self.name).log("node '%d' is an orphan"
                                           % layout.rank)

        self.comm = layout.comm
        self.rank = layout.rank
        self.nodes = layout.nodes

        return



    def reportConfiguration(self):

        rank = self.comm.rank

        if rank != 0:
            return

        self._info.line("configuration:")
#        self._info.line("  properties:")
#        self._info.line("    name: %r" % self.inventory.name)
#        self._info.line("    full name: %r" % self.inventory.fullname)

        self._info.line("  facilities:")
        self._info.line("    journal: %r" % self.inventory.journal.name)
        self._info.line("    launcher: %r" % self.inventory.launcher.name)

        self._info.line("    coarse: %r" % self.inventory.coarse.name)
        self._info.line("    fine: %r" % self.inventory.fine.name)
        self._info.line("    cge: %r" % self.inventory.cge.name)
        self._info.line("    fge: %r" % self.inventory.fge.name)
        self._info.line("    coarseController: %r" % self.inventory.coarseController.name)
        self._info.line("    fineController: %r" % self.inventory.fineController.name)
        self._info.line("    coupler: %r" % self.inventory.coupler.name)
        self._info.line("    layout: %r" % self.inventory.layout.name)

        return



    class Inventory(SimpleApp.Inventory):

        import pyre.inventory

        import Controller
        import Solver
        import Coupler
        import Layout
        import CitcomS.Components.Exchanger as Exchanger


        coarseController = pyre.inventory.facility(name="coarseController", factory=Controller.controller, args=("coarseController",))
        fineController = pyre.inventory.facility(name="fineController", factory=Controller.controller, args=("fineController",))
        coupler = pyre.inventory.facility("coupler", factory=Coupler.coupler)
        layout = pyre.inventory.facility("layout", factory=Layout.layout)

        coarse = pyre.inventory.facility("coarse", factory=Solver.fullSolver, args=("coarse", "coarse"))
        fine = pyre.inventory.facility("fine", factory=Solver.regionalSolver, args=("fine", "fine"))
        cge = pyre.inventory.facility("cge", factory=Exchanger.coarsegridexchanger)
        fge = pyre.inventory.facility("fge", factory=Exchanger.finegridexchanger)

        steps = pyre.inventory.int("steps", default=1)




# main
if __name__ == "__main__":

    app = CoupledApp("CitcomS")
    app.run()



# version
__id__ = "$Id$"

# End of file
