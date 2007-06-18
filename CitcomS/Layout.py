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


from pyre.components.Component import Component


class Layout(Component):


    def __init__(self, name, facility):
        Component.__init__(self, name, facility)

        self.ccomm = None
        self.ecomm = None
        self.ccommPlus = []
        self.ecommPlus = []

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
        if self.nodes < 2:
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
        ccommGroup = self.inventory.ccomm
        ecommGroup = self.inventory.ecomm

        # communicator for solvers
        self.ccomm = world.include(ccommGroup)
        self.ecomm = world.include(ecommGroup)

        # communicator for inter-solver communication
        for node in ccommGroup:
            self.ecommPlus.append(world.include(ecommGroup + [node]))

        for node in ecommGroup:
            self.ccommPlus.append(world.include(ccommGroup + [node]))

        return



    class Inventory(Component.Inventory):

        import pyre.inventory

        ccomm = pyre.inventory.slice("ccomm", default=range(12))
        ecomm = pyre.inventory.slice("ecomm", default=[12])



# version
__id__ = "$Id$"

# End of file
