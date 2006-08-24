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


def layout(name="layout", facility="layout"):
    return Layout(name, facility)


from pyre.components.Component import Component


class Layout(Component):


    def __init__(self, name, facility):
        Component.__init__(self, name, facility)

        self.comm1 = None
        self.comm2 = None
        self.comm1Plus = []
        self.comm2Plus = []

        self.comm = None
        self.rank = 0
        self.nodes = 0
        return



    def initialize(self, application):
        self.discover()
        self.verify(application)
        self.allocateNodes()
        self.createCommunicators()
        return



    def discover(self):
        import mpi
        self.comm = mpi.world()
        self.rank = self.comm.rank
        self.nodes = self.comm.size
        return



    def verify(self, application):
        size = self.nodes
        nodes = application.inventory.launcher.inventory.nodes
        if nodes != size:
            import journal
            firewall = journal.firewall("layout")
            firewall.log("processor count mismatch: %d != %d" % (nodes, size))

        if nodes < 2:
            import journal
            firewall = journal.firewall("layout")
            firewall.log("'%s' requires at least 2 processors"
                         % application.name)

        return



    def allocateNodes(self):
        return



    def createCommunicators(self):
        world = self.comm
        myrank = world.rank
        comm1Group = self.inventory.comm1
        comm2Group = self.inventory.comm2

        # communicator for solvers
        self.comm1 = world.include(comm1Group)
        self.comm2 = world.include(comm2Group)

        # communicator for inter-solver communication
        for node in comm1Group:
            self.comm2Plus.append(world.include(comm2Group + [node]))

        for node in comm2Group:
            self.comm1Plus.append(world.include(comm1Group + [node]))

        return



    class Inventory(Component.Inventory):

        import pyre.inventory

        comm1 = pyre.inventory.slice("comm1", default=range(12))
        comm2 = pyre.inventory.slice("comm2", default=[12])



# version
__id__ = "$Id$"

# End of file
