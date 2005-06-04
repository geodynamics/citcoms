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
        self.coarsePlus = []
        self.finePlus = []

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
        fineGroup = self.inventory.fine
        coarseGroup = self.inventory.coarse

        self.fine = world.include(fineGroup)
        self.coarse = world.include(coarseGroup)

        for each in coarseGroup:
            self.finePlus.append(world.include(fineGroup + [each]))

        for each in fineGroup:
            self.coarsePlus.append(world.include(coarseGroup + [each]))

        return



    class Inventory(Component.Inventory):

        import pyre.inventory


        coarse = pyre.inventory.slice("coarse", default=range(12))
        fine = pyre.inventory.slice("fine", default=[12])



# version
__id__ = "$Id: Layout.py,v 1.11 2005/06/03 21:51:43 leif Exp $"

# End of file
