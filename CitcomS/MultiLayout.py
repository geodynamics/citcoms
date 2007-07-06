#!/usr/bin/env python

#
# Layout for MultiCoupled Application
#
#


from pyre.components.Compenent import Component

class MultiLayout(Component):


    def __init(self, name, facility):
        Compoent.__init__(self, name, facility)

        # flag indicating that we are using
        # containing communicator
        self.ccomm = None

        # flag indicating that we are using
        # embedded comminicator1
        self.ecomm1 = None

        # flag indicating that we are using
        # embedded comminicator2
         self.ecomm2 = None

        # list of communicators created to pass imformation
        # between different solvers
        self.ccomPlus1 = []
        self.ccomPlus2 = []
        self.ecommPlus1 = []
        self.ecommPlus2 = []

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
        #Find the size and rank of the whole application
        import mpi
        self.comm = mpi.world()
        self.rank = self.comm.rank
        self.nodes = self.comm.size
        return

    def verify(self, application):
        # check that we have at least 3 processor
        if self.nodes <3:
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
            self.ccommPlus.append(world.include(containing_group + [node]))
        for node in embedded_group2:
            self.ccommPlus.append(world.include(containing_group + [node]))

        return
       
    class Inventory(Component.Inventory):

        import pyre.inventory

        # The containing solver will run on these nodes
        containing_group = pyre.inventory.slice("containing_group",
                                                default=range(11))

        # The embedded solver1 will run on these nodes
        embedded_group1 = pyre.inventory.slice("embedded_group",
                                              default=[11])

        # The embedded solver2 will run on these nodes
        embedded_group2 = pyre.inventory.slice("embedded_group",
                                              default=[12])


# End of file
