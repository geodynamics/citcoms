#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


def layout(name="layout", facility="layout"):
    return Layout(name, facility)


from pyre.components.Component import Component


class Layout(Component):


    def __init__(self, name, facility):
        Component.__init__(self, name, facility)

        self.coarse = None
        self.fine = None
        self.intercomm = None

        self.rank = 0
        self.nodes = 0
        self.leader = 0
        self.remoteLeader = 0
        return



    def initialize(self, application):

        self.discover()
        self.verify(application)
        self.allocateNodes()
        self.createCommunicators()
        self.setAttributes()
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

        if nodes < 2:
            import journal
            firewall = journal.firewall("layout")
            firewall.log("'%s' requires at least 2 processors" % application.inventory.name)

        return



    def allocateNodes(self):
        return



    def createCommunicators(self):
        import mpi
        world = mpi.world()

        self.fine = world.include(self.inventory.fine)
        self.coarse = world.include(self.inventory.coarse)
        return



    def setAttributes(self):
        if self.fine:
            mygroup = self.inventory.fine
            remotegroup = self.inventory.coarse
            self.createIntercomm(self.fine)
        elif self.coarse:
            mygroup = self.inventory.coarse
            remotegroup = self.inventory.fine
            self.createIntercomm(self.coarse)
        else:
            import journal
            journal.warning(self.name).log("node '%d' is an orphan" % self.rank)

        # use the last proc. as the group leader
        self.leader = len(mygroup) - 1
        self.remoteLeader = remotegroup[-1]

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

            ]


# version
__id__ = "$Id: Layout.py,v 1.9 2003/10/24 04:55:54 tan2 Exp $"

# End of file
