#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.components.Component import Component


class Layout(Component):


    def __init__(self, name, facility):
        Component.__init__(self, name, facility)

        self.coarse = None
        self.fine = None
        self.intercomm = None

        self.rank = 0
        self.nodes = 0
        self.localLeader = 0
        self.remoteLeader = 0
        return



    def initialize(self, application):

        self.discover()
        self.verify(application)
        self.allocateNodes()
        self.createCommunicators()

        return



    def discover(self):
        import mpi
        self.rank = mpi.world().rank
        self.nodes = mpi.world().size
        return



    def verify(self, application):

        size = self.nodes
        nodes = application.inventory.staging.inventory.nodes
        if nodes != size:
            import journal
            firewall = journal.firewall("layout")
            firewall.log("processor count mismatch: %d != %d" % (nodes, size))

        return
        if nodes < 2:
            import journal
            firewall = journal.firewall("layout")
            firewall.log("'%s' requires at least 2 processors" % application.inventory.name)

        return



    def allocateNodes(self):
        rank = self.rank
        nodes = self.nodes
        #ratio = self.inventory.ratio

        fine = self.inventory.fine
        coarse = self.inventory.coarse

        #sr, fr = map(int, ratio.split(":"))

        #sp = int(nodes * sr/(sr+fr))
        #fp = nodes - sp

        #if len(fine) < sp or len(coarse) < fp:
        #    fine = range(sp)
        #    coarse = range(sp, nodes)

        #self.inventory.fine = fine
        #self.inventory.coarse = coarse
        return



    def createCommunicators(self):
        import mpi
        world = mpi.world()

        self.fine = world.include(self.inventory.fine)
        self.coarse = world.include(self.inventory.coarse)

        if self.fine:
            # use the last proc. as the group leader
            self.leaderRank = len(self.inventory.fine) - 1 
            self.localLeader = self.inventory.fine[-1]
            self.remoteLeader = self.inventory.coarse[-1]
            self.createIntercomm(self.fine)
        elif self.coarse:
            # use the last proc. as the group leader
            self.leaderRank = len(self.inventory.coarse) - 1
            self.localLeader = self.inventory.coarse[-1]
            self.remoteLeader = self.inventory.fine[-1]
            self.createIntercomm(self.coarse)
        else:
            import journal
            journal.warning(self.name).log("node '%d' is an orphan" % self.rank)

        return



    def createIntercomm(self, comm):
        # not finished
        import mpi
        self.intercomm = mpi.world()
        return



    class Inventory(Component.Inventory):

        import pyre.properties

        inventory = [

            pyre.properties.sequence("coarse", range(12)),
            pyre.properties.sequence("fine", [12]),

            # test
            #pyre.properties.sequence("coarse", [0]),
            #pyre.properties.sequence("fine", [1]),

            ]


# version
__id__ = "$Id: Layout.py,v 1.6 2003/09/27 20:17:51 tan2 Exp $"

# End of file
