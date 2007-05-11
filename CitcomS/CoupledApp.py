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

from BaseApplication import BaseApplication
import journal


class CoupledApp(BaseApplication):


    def __init__(self, name="CoupledCitcomS"):
        BaseApplication.__init__(self, name)

        self.solver = None
        self.solverCommunicator = None
        self.myPlus = []
        self.remotePlus = []

        self.comm = None
        self.rank = 0
        self.nodes = 0
        return



    def getNodes(self):
        s1 = self.inventory.solver1.inventory.mesher.inventory
        nproc1 = s1.nproc_surf * s1.nprocx * s1.nprocy * s1.nprocz
        s2 = self.inventory.solver2.inventory.mesher.inventory
        nproc2 = s2.nproc_surf * s2.nprocx * s2.nprocy * s2.nprocz
        return nproc1 + nproc2



    def initialize(self):
        layout = self.inventory.layout
        layout.initialize(self)

        self.findLayout(layout)

        self.controller.initialize(self)

        return



    def findLayout(self, layout):

        if layout.comm1:
            self.controller = self.inventory.controller1
            self.solver = self.inventory.solver1
            self.coupler = self.inventory.coupler1
            self.solverCommunicator = layout.comm1
            self.myPlus = layout.comm1Plus
            self.remotePlus = layout.comm2Plus
        elif layout.comm2:
            self.controller = self.inventory.controller2
            self.solver = self.inventory.solver2
            self.coupler = self.inventory.coupler2
            self.solverCommunicator = layout.comm2
            self.myPlus = layout.comm2Plus
            self.remotePlus = layout.comm1Plus
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
        self._info.line("    launcher: %r" % self.inventory.launcher.name)

        self._info.line("    solver1: %r" % self.inventory.solver1.name)
        self._info.line("    solver2: %r" % self.inventory.solver2.name)
        self._info.line("    controller1: %r" % self.inventory.controller1.name)
        self._info.line("    controller2: %r" % self.inventory.controller2.name)
        self._info.line("    coupler1: %r" % self.inventory.coupler1.name)
        self._info.line("    coupler2: %r" % self.inventory.coupler2.name)
        self._info.line("    layout: %r" % self.inventory.layout.name)

        return



    class Inventory(BaseApplication.Inventory):

        import pyre.inventory

        import Controller
        import Solver
        import Coupler
        import Layout

        controller1 = pyre.inventory.facility(name="controller1",
                                              factory=Controller.controller,
                                              args=("ccontroller","controller1"))
        controller2 = pyre.inventory.facility(name="controller2",
                                              factory=Controller.controller,
                                              args=("econtroller","controller2"))
        coupler1 = pyre.inventory.facility("coupler1",
                                           factory=Coupler.containingcoupler,
                                           args=("ccoupler","coupler1"))
        coupler2 = pyre.inventory.facility("coupler2",
                                           factory=Coupler.embeddedcoupler,
                                           args=("ecoupler","coupler2"))

        solver1 = pyre.inventory.facility("solver1",
                                          factory=Solver.coupledRegionalSolver,
                                          args=("csolver", "solver1"))
        solver2 = pyre.inventory.facility("solver2",
                                       factory=Solver.coupledRegionalSolver,
                                       args=("esolver", "solver2"))

        layout = pyre.inventory.facility("layout", factory=Layout.layout)

        steps = pyre.inventory.int("steps", default=1)




# version
__id__ = "$Id$"

# End of file
