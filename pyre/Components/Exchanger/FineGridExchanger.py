#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
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
        self.globalBBox = self.module.createGlobalBoundedBox(self.all_variables)
        mycomm = self.communicator
        self.remoteBBox = self.module.exchangeBoundedBox(self.globalBBox,
                                                         mycomm.handle(),
                                                         self.sinkComm.handle(),
                                                         0)
        self.boundary, self.myBBox = self.module.createBoundary(
                                                     self.remoteBBox,
                                                     self.all_variables)
        for i in range(len(self.interior)):
            self.interior[i] = self.module.createEmptyInterior()

        return


    def createSourceSink(self):
        self.createSink()
        self.createSource()
        return


    def createSink(self):
        self.sink["BC"] = self.module.createSink(self.sinkComm.handle(),
                                                 self.numSrc,
                                                 self.boundary)
        return


    def createSource(self):
        for i, comm, b in zip(range(self.numSrc),
                              self.srcComm,
                              self.interior):
            # sink is always in the last rank of a communicator
            sinkRank = comm.size - 1
            self.source["Intr"][i] = self.module.VTSource_create(
                                                     comm.handle(),
                                                     sinkRank,
                                                     b,
                                                     self.all_variables,
                                                     self.myBBox)
        return


    def createBC(self):
        if self.inventory.incompressibility:
            self.BC = self.module.BoundaryVTInlet_create(self.communicator.handle(),
                                                         self.boundary,
                                                         self.sink["BC"],
                                                         self.all_variables,
                                                         "VT")
        else:
            self.BC = self.module.VTInlet_create(self.boundary,
                                                 self.sink["BC"],
                                                 self.all_variables,
                                                 "VT")

        return


    def createII(self):
        for i, src in zip(range(self.numSrc),
                          self.source["Intr"]):
            self.II[i] = self.module.VTOutlet_create(src,
                                                     self.all_variables,
                                                     "T")
        return


    def initTemperature(self):
        if self.restart:
            # receive temperature from CGE and postprocess
            self.restartTemperature()
        else:
            self.module.initTemperatureTest(self.globalBBox,
                                            self.all_variables)
        return


    def restartTemperature(self):
        interior, bbox = self.module.createInterior(self.remoteBBox,
                                                    self.all_variables)
        sink = self.module.createSink(self.sinkComm.handle(),
                                      self.numSrc,
                                      interior)
        self.module.initTemperatureSink(interior, sink, self.all_variables)

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
        if self.catchup:
            # send temperture field to CGE
            for ii in self.II:
                self.module.VTOutlet_send(ii)

        return


    def applyBoundaryConditions(self):
        if self.toApplyBC:
            if self.inventory.incompressibility:
                self.module.BoundaryVTInlet_recv(self.BC)
            else:
                self.module.VTInlet_recv(self.BC)

            self.toApplyBC = False

        if self.inventory.incompressibility:
            self.module.BoundaryVTInlet_impose(self.BC)
        else:
            self.module.VTInlet_impose(self.BC)

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
        if self.inventory.incompressibility:
            self.module.BoundaryVTInlet_storeTimestep(self.BC, self.fge_t, self.cge_t)
        else:
            self.module.VTInlet_storeTimestep(self.BC, self.fge_t, self.cge_t)

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

        import pyre.properties as prop


        inventory = [

            prop.bool("incompressibility", default=True),

            ]



# version
__id__ = "$Id: FineGridExchanger.py,v 1.32 2004/02/25 00:54:42 tan2 Exp $"

# End of file
