#!/usr/bin/env python

#
# Layout for MultiCoupled Application
#
#


from Layout import Layout

class MultiLayout(Layout):


    def __init__(self, name, facility):
        Layout.__init__(self, name, facility)

        # containing communicator
        self.ccomm = None

        # embedded comminicator1
        self.ecomm1 = None

        # embedded comminicator2
        self.ecomm2 = None

        # list of communicators created to pass imformation
        # between different solvers
        self.ccommPlus1 = []
        self.ccommPlus2 = []
        self.ecommPlus1 = []
        self.ecommPlus2 = []

        self.comm = None
        self.rank = 0
        self.nodes = 0
        return


    def verify(self, application):
        # check that we have at least 3 processor
        if self.nodes < 3:
            import journal
            firewall = journal.firewall("MultiLayout")
            firewall.log("'%s' requires at least 3 processors" \
                         % application.name)


        containing_group = self.inventory.containing_group
        embedded_group1 = self.inventory.embedded_group1
        embedded_group2 = self.inventory.embedded_group2

        # check for duplicated elements in the group
        self.check_duplicated(containing_group)
        self.check_duplicated(embedded_group1)
        self.check_duplicated(embedded_group2)

        # check that the three groups are disjoint
        self.check_disjoint(containing_group, embedded_group1)
        self.check_disjoint(containing_group, embedded_group2)
        self.check_disjoint(embedded_group1, embedded_group2)

        return



    def createCommunicators(self):
        # Create communicators for solvers and couplerd

        world = self.comm
        myrank = self.rank
        containing_group = self.inventory.containing_group
        embedded_group1 = self.inventory.embedded_group1
        embedded_group2 = self.inventory.embedded_group2

        # Communicator for solvers
        self.ccomm = world.include(containing_group)
        self.ecomm1 = world.include(embedded_group1)
        self.ecomm2 = world.include(embedded_group2)

        # Communicator for inter-solver communication
        # ecommPlus1 is a list of communicators, with each communicator
        # contains a node in embedded_group1 and the whole containing_group

        # ecommPlus2 is similar
        for node in containing_group:
            self.ecommPlus1.append(world.include(embedded_group1 + [node]))
            self.ecommPlus2.append(world.include(embedded_group2 + [node]))

        # ccommPlus1 is a list of communicators, with each communicator
        # contains a node in containing group and the whole embedded_group1

        # commPlus2 is similar
        for node in embedded_group1:
            self.ccommPlus1.append(world.include(containing_group + [node]))
        for node in embedded_group2:
            self.ccommPlus2.append(world.include(containing_group + [node]))

        return


    class Inventory(Layout.Inventory):

        import pyre.inventory

        # The containing solver will run on these nodes
        containing_group = pyre.inventory.slice("containing_group",
                                                default=range(12))

        # The embedded solver1 will run on these nodes
        embedded_group1 = pyre.inventory.slice("embedded_group1",
                                               default=[12])

        # The embedded solver2 will run on these nodes
        embedded_group2 = pyre.inventory.slice("embedded_group2",
                                               default=[13])


# version
__id__ = "$Id: MultiLayout.py 7674 2007-07-16 21:21:10Z hlin $"

# End of file
