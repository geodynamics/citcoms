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

from Exchanger import Exchanger

class FineGridExchanger(Exchanger):


    def __init__(self, name, facility):
        Exchanger.__init__(self, name, facility)
        self.cge_t = 0
        self.fge_t = 0
        self.toApplyBC = True

        # exchanged information is non-dimensional
        self.inventory.dimensional = False
        # exchanged information is in spherical coordinate
        self.inventory.transformational = False
        return


    def initialize(self, solver):
        Exchanger.initialize(self, solver)

	# restart and use temperautre field of previous run?
        self.restart = solver.restart
        if self.restart:
            self.ic_initTemperature = solver.ic_initTemperature

	self.all_variables = solver.all_variables
        self.interior = range(self.numSrc)
        self.source["Intr"] = range(self.numSrc)
        self.II = range(self.numSrc)

        self.module.initConvertor(self.inventory.dimensional,
                                  self.inventory.transformational,
                                  self.all_variables)

        return


    def createMesh(self):
        inv = self.inventory
        self.globalBBox = self.module.createGlobalBoundedBox(self.all_variables)
        mycomm = self.communicator
        self.remoteBBox = self.module.exchangeBoundedBox(self.globalBBox,
                                                         mycomm.handle(),
                                                         self.sinkComm.handle(),
                                                         0)
        self.boundary, self.myBBox = self.module.createBoundary(
                                                     self.all_variables,
                                                     inv.excludeTop,
                                                     inv.excludeBottom)

        if inv.two_way_communication:
            for i in range(len(self.interior)):
                self.interior[i] = self.module.createEmptyInterior()

        return


    def createSourceSink(self):
        self.createSink()

        if self.inventory.two_way_communication:
            self.createSource()
        return


    def createSink(self):
        self.sink["BC"] = self.module.Sink_create(self.sinkComm.handle(),
                                                  self.numSrc,
                                                  self.boundary)
        return


    def createSource(self):
        for i, comm, b in zip(range(self.numSrc),
                              self.srcComm,
                              self.interior):
            # sink is always in the last rank of a communicator
            sinkRank = comm.size - 1
            self.source["Intr"][i] = self.module.CitcomSource_create(
                                                     comm.handle(),
                                                     sinkRank,
                                                     b,
                                                     self.myBBox,
                                                     self.all_variables)

        return


    def createBC(self):
        import Inlet
        self.BC = Inlet.SVTInlet(self.boundary,
                                 self.sink["BC"],
                                 self.all_variables)
        '''
        if self.inventory.incompressibility:
            self.BC = Inlet.BoundaryVTInlet(self.communicator,
                                            self.boundary,
                                            self.sink["BC"],
                                            self.all_variables,
                                            "VT")
            import journal
            journal.info("incompressibility").activate()
        else:
            self.BC = Inlet.SVTInlet(self.boundary,
                                    self.sink["BC"],
                                    self.all_variables)
        '''
        return


    def createII(self):
        import Outlet
        for i, src in zip(range(self.numSrc),
                          self.source["Intr"]):
            self.II[i] = Outlet.TOutlet(src,
                                        self.all_variables)
        return


    def initTemperature(self):
        if self.restart:
            # receive temperature from CGE and postprocess
            self.restartTemperature()
        else:
            self.module.initTemperature(self.globalBBox,
                                        self.all_variables)
        return


    def restartTemperature(self):
        interior, bbox = self.module.createInterior(self.remoteBBox,
                                                    self.all_variables)
        sink = self.module.Sink_create(self.sinkComm.handle(),
                                       self.numSrc,
                                       interior)
        import Inlet
        inlet = Inlet.TInlet(interior, sink, self.all_variables)
        inlet.recv()
        inlet.impose()

        # Any modification of read-in temperature is done here
        # Note: modifyT is called after receiving unmodified T from CGE.
        # If T is modified before sending, FGE's T will lose sharp feature.
        # CGE has to call modifyT too to ensure consistent T field.
        self.modifyT(self.globalBBox)

        return


    def preVSolverRun(self):
        self.applyBoundaryConditions()
        return


    def NewStep(self):
        if self.inventory.two_way_communication:
            if self.catchup:
                # send temperture field to CGE
                for ii in self.II:
                    ii.send()

        return


    def applyBoundaryConditions(self):
        if self.toApplyBC:
            self.BC.recv()

            self.toApplyBC = False

        self.BC.impose()

        # applyBC only when previous step is a catchup step
        if self.catchup:
            self.toApplyBC = True

        return


    def stableTimestep(self, dt):
        if self.catchup:
            mycomm = self.communicator
            self.cge_t = self.module.exchangeTimestep(dt,
                                                      mycomm.handle(),
                                                      self.sinkComm.handle(),
                                                      0)
            self.fge_t = 0
            self.catchup = False

        self.fge_t += dt
        old_dt = dt

        if self.fge_t >= self.cge_t:
            dt = dt - (self.fge_t - self.cge_t)
            self.fge_t = self.cge_t
            self.catchup = True
            #print "FGE: CATCHUP!"

        # store timestep for interpolating boundary velocities
        self.BC.storeTimestep(self.fge_t, self.cge_t)

        #print "%s - old dt = %g   exchanged dt = %g" % (
        #       self.__class__, old_dt, dt)
        #print "cge_t = %g  fge_t = %g" % (self.cge_t, self.fge_t)
        return dt


    def exchangeSignal(self, signal):
        mycomm = self.communicator
        newsgnl = self.module.exchangeSignal(signal,
                                             mycomm.handle(),
                                             self.sinkComm.handle(),
                                             0)
        return newsgnl



    class Inventory(Exchanger.Inventory):

        import pyre.inventory as prop



        excludeTop = prop.bool("excludeTop", default=False)
        excludeBottom = prop.bool("excludeBottom", default=False)
        incompressibility = prop.bool("incompressibility", default=True)




# version
__id__ = "$Id: FineGridExchanger.py,v 1.41 2005/06/10 02:23:22 leif Exp $"

# End of file
