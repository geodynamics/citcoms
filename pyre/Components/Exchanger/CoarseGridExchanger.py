#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from Exchanger import Exchanger

class CoarseGridExchanger(Exchanger):


    def __init__(self, name, facility):
        Exchanger.__init__(self, name, facility)

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
        self.boundary = range(self.numSrc)
        self.source["BC"] = range(self.numSrc)
        self.BC = range(self.numSrc)

        self.module.initConvertor(self.inventory.dimensional,
                                  self.inventory.transformational,
                                  self.all_variables)

        return


    def createMesh(self):
        self.globalBBox = self.module.createGlobalBoundedBox(self.all_variables)
        self.remoteBBox = self.module.exchangeBoundedBox(
                                          self.globalBBox,
                                          self.communicator.handle(),
                                          self.srcComm[0].handle(),
                                          self.srcComm[0].size - 1)
        self.interior, self.myBBox = self.module.createInterior(
                                                     self.remoteBBox,
                                                     self.all_variables)
        for i in range(len(self.boundary)):
            self.boundary[i] = self.module.createEmptyBoundary()

        return


    def createSourceSink(self):
        self.createSource()

        if self.inventory.two_way_communication:
            self.createSink()
        return


    def createSource(self):
        for i, comm, b in zip(range(self.numSrc),
                              self.srcComm,
                              self.boundary):
            # sink is always in the last rank of a communicator
            sinkRank = comm.size - 1
            self.source["BC"][i] = self.module.CitcomSource_create(comm.handle(),
                                                                   sinkRank,
                                                                   b,
                                                                   self.myBBox,
                                                                   self.all_variables)

        return


    def createSink(self):
        self.sink["Intr"] = self.module.Sink_create(self.sinkComm.handle(),
                                                    self.numSrc,
                                                    self.interior)
        return


    def createBC(self):
        import Outlet
        for i, src in zip(range(self.numSrc),
                          self.source["BC"]):
            self.BC[i] = Outlet.SVTOutlet(src,
                                          self.all_variables)
        return


    def createII(self):
        import Inlet
        self.II = Inlet.VTInlet(self.interior,
                                self.sink["Intr"],
                                self.all_variables)
        return


    def initTemperature(self):
        if self.restart:
            # read-in restarted temperature field
            self.ic_initTemperature()
            del self.ic_initTemperature
            # send temperature to FGE and postprocess
            self.restartTemperature()
        else:
            self.module.initTemperature(self.remoteBBox,
                                        self.all_variables)
        return


    def restartTemperature(self):
        interior = range(self.numSrc)
        source = range(self.numSrc)

        for i in range(len(interior)):
            interior[i] = self.module.createEmptyInterior()

        for i, comm, b in zip(range(self.numSrc),
                              self.srcComm,
                              interior):
            # sink is always in the last rank of a communicator
            sinkRank = comm.size - 1
            source[i] = self.module.CitcomSource_create(comm.handle(),
                                                        sinkRank,
                                                        b,
                                                        self.myBBox,
                                                        self.all_variables)

        import Outlet
        for i, src in zip(range(self.numSrc), source):
            outlet = Outlet.TOutlet(src, self.all_variables)
            outlet.send()

        # Any modification of read-in temperature is done here
        # Note: modifyT is called after sending unmodified T to FGE.
        # If T is modified before sending, FGE's T will lose sharp feature.
        # FGE has to call modifyT too to ensure consistent T field.
        self.modifyT(self.remoteBBox)

        return


    def postVSolverRun(self):
        self.applyBoundaryConditions()
        return


    def NewStep(self):
        if self.inventory.two_way_communication:
            # receive temperture field from FGE
            self.II.recv()
            self.II.impose()
        return


    def applyBoundaryConditions(self):
        for bc in self.BC:
            bc.send()
        return


    def stableTimestep(self, dt):
        new_dt = self.module.exchangeTimestep(dt,
                                              self.communicator.handle(),
                                              self.srcComm[0].handle(),
                                              self.srcComm[0].size - 1)
        #print "%s - old dt = %g   exchanged dt = %g" % (
        #       self.__class__, dt, new_dt)
        return dt


    def exchangeSignal(self, signal):
        newsgnl = self.module.exchangeSignal(signal,
                                             self.communicator.handle(),
                                             self.srcComm[0].handle(),
                                             self.srcComm[0].size - 1)
        return newsgnl



    class Inventory(Exchanger.Inventory):

        import pyre.properties as prop


        inventory = [

            ]



# version
__id__ = "$Id: CoarseGridExchanger.py,v 1.35 2004/05/29 01:19:31 tan2 Exp $"

# End of file
