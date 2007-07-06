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
        # csolver requires nproc1 CPUs to run
        s1 = self.inventory.csolver.inventory.mesher.inventory
        nproc1 = s1.nproc_surf * s1.nprocx * s1.nprocy * s1.nprocz

        # esolver requires nproc2 CPUs to run
        s2 = self.inventory.esolver.inventory.mesher.inventory
        nproc2 = s2.nproc_surf * s2.nprocx * s2.nprocy * s2.nprocz

        # the whole application requires nproc1+nproc2 CPUs
        return nproc1 + nproc2



    def initialize(self):
        layout = self.inventory.layout
        layout.initialize(self)

        self.findLayout(layout)

        self.controller.initialize(self)

        return



    def findLayout(self, layout):
        '''Assigning controller/solver/coupler/communicator to this processor.
        '''
        if layout.ccomm:
            # This process belongs to the containing solver
            self.controller = self.inventory.ccontroller
            self.solver = self.inventory.csolver
            self.coupler = self.inventory.ccoupler
            self.solverCommunicator = layout.ccomm
            self.myPlus = layout.ccommPlus
            self.remotePlus = layout.ecommPlus
        elif layout.ecomm:
            # This process belongs to the embedded solver
            self.controller = self.inventory.econtroller
            self.solver = self.inventory.esolver
            self.coupler = self.inventory.ecoupler
            self.solverCommunicator = layout.ecomm
            self.myPlus = layout.ecommPlus
            self.remotePlus = layout.ccommPlus
        else:
            # This process doesn't belong to any solver
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

        self._info.line("    csolver: %r" % self.inventory.csolver.name)
        self._info.line("    esolver: %r" % self.inventory.esolver.name)
        self._info.line("    ccontroller: %r" % self.inventory.ccontroller.name)
        self._info.line("    econtroller: %r" % self.inventory.econtroller.name)
        self._info.line("    ccoupler: %r" % self.inventory.ccoupler.name)
        self._info.line("    ecoupler: %r" % self.inventory.ecoupler.name)
        self._info.log("    layout: %r" % self.inventory.layout.name)

        return



    class Inventory(BaseApplication.Inventory):

        import pyre.inventory

        import Controller
        import Solver
        import Coupler
        import Layout

        ccontroller = pyre.inventory.facility(name="ccontroller",
                                              factory=Controller.controller,
                                              args=("ccontroller","ccontroller"))
        econtroller = pyre.inventory.facility(name="econtroller",
                                              factory=Controller.controller,
                                              args=("econtroller","econtroller"))
        ccoupler = pyre.inventory.facility("ccoupler",
                                           factory=Coupler.containingcoupler,
                                           args=("ccoupler","ccoupler"))
        ecoupler = pyre.inventory.facility("ecoupler",
                                           factory=Coupler.embeddedcoupler,
                                           args=("ecoupler","ecoupler"))

        csolver = pyre.inventory.facility("csolver",
                                          factory=Solver.coupledFullSolver,
                                          args=("csolver", "csolver"))
        esolver = pyre.inventory.facility("esolver",
                                       factory=Solver.coupledRegionalSolver,
                                       args=("esolver", "esolver"))

        layout = pyre.inventory.facility("layout", factory=Layout.Layout,
                                         args=("layout", "layout"))

        steps = pyre.inventory.int("steps", default=1)




# version
__id__ = "$Id$"

# End of file
