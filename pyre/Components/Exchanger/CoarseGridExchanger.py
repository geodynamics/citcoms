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


    def initialize(self, solver):
        Exchanger.initialize(self, solver)

        self.boundary = range(self.numSrc)
        self.source["BC"] = range(self.numSrc)
        self.BC = range(self.numSrc)
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
        self.createSink()
        return


    def createSource(self):
        for i, comm, b in zip(range(self.numSrc),
                              self.srcComm,
                              self.boundary):
            # sink is always in the last rank of a communicator
            self.source["BC"][i] = self.module.createSource(comm.handle(),
                                                            comm.size - 1,
                                                            b,
                                                            self.all_variables,
                                                            self.myBBox)
        return


    def createSink(self):
        return


    def createBC(self):
        for i, src in zip(range(self.numSrc),
                          self.source["BC"]):
            self.BC[i] = self.module.createBCSource(src,
                                                    self.all_variables)
        return


    def initTemperature(self):
        self.module.initTemperatureTest(self.remoteBBox, self.all_variables)
        # send temperture field to FGE
        #self.module.sendTemperature(self.exchanger)
        return


    def postVSolverRun(self):
        self.applyBoundaryConditions()
        return


    def NewStep(self):
        # receive temperture field from FGE
        #self.module.receiveTemperature(self.exchanger)
        return


    def applyBoundaryConditions(self):
        for bc in self.BC:
            self.module.sendTandV(bc)
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
__id__ = "$Id: CoarseGridExchanger.py,v 1.20 2003/11/07 01:08:22 tan2 Exp $"

# End of file
