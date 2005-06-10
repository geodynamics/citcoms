#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#=====================================================================
#
#                             CitcomS.py
#                 ---------------------------------
#
#                              Authors:
#            Eh Tan, Eun-seo Choi, and Pururav Thoutireddy 
#          (c) California Institute of Technology 2002-2005
#
#        By downloading and/or installing this software you have
#       agreed to the CitcomS.py-LICENSE bundled with this software.
#             Free for non-commercial academic research ONLY.
#      This program is distributed WITHOUT ANY WARRANTY whatsoever.
#
#=====================================================================
#
#  Copyright June 2005, by the California Institute of Technology.
#  ALL RIGHTS RESERVED. United States Government Sponsorship Acknowledged.
# 
#  Any commercial use must be negotiated with the Office of Technology
#  Transfer at the California Institute of Technology. This software
#  may be subject to U.S. export control laws and regulations. By
#  accepting this software, the user agrees to comply with all
#  applicable U.S. export laws and regulations, including the
#  International Traffic and Arms Regulations, 22 C.F.R. 120-130 and
#  the Export Administration Regulations, 15 C.F.R. 730-744. User has
#  the responsibility to obtain export licenses, or other export
#  authority as may be required before exporting such information to
#  foreign countries or providing access to foreign nationals.  In no
#  event shall the California Institute of Technology be liable to any
#  party for direct, indirect, special, incidental or consequential
#  damages, including lost profits, arising out of the use of this
#  software and its documentation, even if the California Institute of
#  Technology has been advised of the possibility of such damage.
# 
#  The California Institute of Technology specifically disclaims any
#  warranties, including the implied warranties or merchantability and
#  fitness for a particular purpose. The software and documentation
#  provided hereunder is on an "as is" basis, and the California
#  Institute of Technology has no obligations to provide maintenance,
#  support, updates, enhancements or modifications.
#
#=====================================================================
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
__id__ = "$Id: Layout.py,v 1.12 2005/06/10 02:23:20 leif Exp $"

# End of file
