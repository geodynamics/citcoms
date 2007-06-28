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
        self.createCommunicators()
        return



    def discover(self):
        '''Find the size/rank of the whole application.
        '''
        import mpi
        self.comm = mpi.world()
        self.rank = self.comm.rank
        self.nodes = self.comm.size
        return



    def verify(self, application):
        # check do we have at least 2 processor
        if self.nodes < 2:
            import journal
            firewall = journal.firewall("layout")
            firewall.log("'%s' requires at least 2 processors"
                         % application.name)

        containing_group = self.inventory.containing_group
        embedded_group = self.inventory.embedded_group

        # check no duplicated elements in the group
        self.check_duplicated(containing_group)
        self.check_duplicated(embedded_group)

        # check the two groups are disjoint
        self.check_disjoint(containing_group, embedded_group)

        return



    def check_duplicated(self, group):
        s = set(group)
        if len(s) != len(group):
            import journal
            firewall = journal.firewall("layout")
            firewall.log('Duplicated element in group: %s' % group)
        return



    def check_disjoint(self, group0, group1):
        s0 = set(group0)
        s1 = set(group1)
        if s0.intersection(s1):
            import journal
            firewall = journal.firewall("layout")
            firewall.log('Groups are not disjoint: %s and %s' % (group0, group1))
        return



    def createCommunicators(self):
        '''Create various communicators for solvers and couplers
        '''
        world = self.comm
        myrank = self.rank
        containing_group = self.inventory.containing_group
        embedded_group = self.inventory.embedded_group

        # Communicator for solvers
        self.ccomm = world.include(containing_group)
        self.ecomm = world.include(embedded_group)

        # Communicator for inter-solver communication
        # Each node in containing_group will form a communicator
        # with the nodes in embedded_group
        for node in containing_group:
            self.ecommPlus.append(world.include(embedded_group + [node]))

        # Ditto for each node containing_group
        for node in embedded_group:
            self.ccommPlus.append(world.include(containing_group + [node]))

        return



    class Inventory(Component.Inventory):

        import pyre.inventory

        # The containing solver will run on these nodes
        containing_group = pyre.inventory.slice("containing_group",
                                                default=range(12))

        # The embedded solver will run on these nodes
        embedded_group = pyre.inventory.slice("embedded_group",
                                              default=[12])



# version
__id__ = "$Id$"

# End of file
