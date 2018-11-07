#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#
# CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
# Copyright (C) 2002-2005, California Institute of Technology.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
#</LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from Coupler import Coupler

class EmbeddedCoupler(Coupler):


    def __init__(self, name, facility):
        Coupler.__init__(self, name, facility)

        # time of containing coupler
        self.ccplr_t = 0

        # time of embedded coupler
        self.ecplr_t = 0

        # whether to apply boundary conditions
        self.toApplyBC = True
        return


    def initialize(self, solver):
        Coupler.initialize(self, solver)

        # the embedded solver should set its solver.bc.side_sbcs to on
        # otherwise, we have to stop
        assert solver.inventory.bc.inventory.side_sbcs == True, \
               'Error: esolver.bc.side_sbcs must be on!'

        # XXX: model/Exchanger/AreaWeightedNormal.cc
        # computeOutflow() and reduceOutflow() needs to be modified
        # to support compressible flow
        assert (not self.inventory.amending_outflow) or \
               solver.inventory.gruneisen == 0, \
               'Error, amending_outflow is not implemented for compressible flow!'
        return


    def createMesh(self):
        '''Create BoundedMesh objects.
        '''
        from ExchangerLib import createGlobalBoundedBox, exchangeBoundedBox, createBoundary, createEmptyInterior
        inv = self.inventory

        # the bounding box of the mesh on this solver
        self.globalBBox = createGlobalBoundedBox(self.all_variables)

        # the bounding box of the mesh on the other solver
        mycomm = self.communicator
        self.remoteBBox = exchangeBoundedBox(self.globalBBox,
                                             mycomm.handle(),
                                             self.sinkComm.handle(),
                                             0)

        # the nodes on the boundary, top and bottom boundaries are special
        self.boundary, self.myBBox = createBoundary(self.all_variables,
                                                    inv.exclude_top,
                                                    inv.exclude_bottom)

        # an empty interior object, which will be filled by a remote interior obj.
        if inv.two_way_communication:
            self.remoteIntrList = range(self.remoteSize)
            for i in range(self.remoteSize):
                self.remoteIntrList[i] = createEmptyInterior()

        return


    def createSourceSink(self):
        # create sink first, then source. The order is important.
        self.createSink()

        if self.inventory.two_way_communication:
            self.createSource()
        return


    def createSink(self):
        # the sink obj. will receive boundary conditions from remote sources
        from ExchangerLib import Sink_create
        # the sink will communicate with the source in ContainingCoupler
        # during creation stage
        self.sink = Sink_create(self.sinkComm.handle(),
                                self.remoteSize,
                                self.boundary)

        if self.inventory.exchange_pressure:
            from ExchangerLib import createPInterior
            self.pinterior, _ = createPInterior(self.myBBox,
                                                self.all_variables)
            self.psink = Sink_create(self.sinkComm.handle(),
                                     self.remoteSize,
                                     self.pinterior)
        return


    def createSource(self):
        # the source obj's will send interior temperature to a remote sink
        from ExchangerLib import CitcomSource_create
        self.sourceList = range(self.remoteSize)
        for i, comm, b in zip(range(self.remoteSize),
                              self.srcCommList,
                              self.remoteIntrList):
            # sink is always in the last rank of a communicator
            sinkRank = comm.size - 1

            # the sources will communicate with the sink in ContainingCoupler
            # during creation stage
            self.sourceList[i] = CitcomSource_create(comm.handle(),
                                                     sinkRank,
                                                     b,
                                                     self.myBBox,
                                                     self.all_variables)

        return


    def createBC(self):
        # boundary conditions will be recv. by SVTInlet, which receives
        # stress, velocity, and temperature
        import Inlet
        if self.inventory.amending_outflow:
            self.inlet = Inlet.BoundarySVTInlet(self.boundary,
                                                self.sink,
                                                self.all_variables,
                                                self.communicator.handle())
        else:
            self.inlet = Inlet.SVTInlet(self.boundary,
                                        self.sink,
                                        self.all_variables)

        if self.inventory.exchange_pressure:
            self.pinlet = Inlet.PInlet(self.pinterior,
                                       self.psink,
                                       self.all_variables)
        return


    def createII(self):
        # interior temperature will be sent by TOutlet
        import Outlet
        self.outletList = range(self.remoteSize)
        for i, src in zip(range(self.remoteSize),
                          self.sourceList):
            self.outletList[i] = Outlet.TOutlet(src, self.all_variables)
        return


    def initTemperature(self):
        from ExchangerLib import initTemperature
        initTemperature(self.globalBBox,
                        self.all_variables)
        return


    def exchangeTemperature(self):
        if not self.inventory.exchange_initial_temperature:
            return

        from ExchangerLib import createInterior, Sink_create
        interior, bbox = createInterior(self.remoteBBox,
                                        self.all_variables)
        sink = Sink_create(self.sinkComm.handle(),
                           self.remoteSize,
                           interior)
        import Inlet
        inlet = Inlet.TInlet(interior, sink, self.all_variables)
        inlet.recv()
        inlet.impose()

        # Any modification of read-in temperature is done here
        # Note: modifyT is called after receiving unmodified T from CCPLR.
        # If T is modified before sending, ECPLR's T will lose sharp feature.
        # CCPLR has to call modifyT too to ensure consistent T field.
        #self.modifyT(self.globalBBox)

        return


    def preVSolverRun(self):
        # apply bc before solving the velocity

        # recv'ing vbc after sync'd steps
        if self.toApplyBC:
            self.inlet.recv()

            self.toApplyBC = False

            if self.inventory.exchange_pressure:
                self.pinlet.recv()
                self.pinlet.impose()

        # update vbc every time step
        self.inlet.impose()

        # applyBC only when previous step is sync'd
        if self.synchronized:
            self.toApplyBC = True
        return


    def newStep(self):
        if self.inventory.two_way_communication:
            if self.synchronized:
                # send temperture field to CCPLR
                for outlet in self.outletList:
                    outlet.send()

        return



    def stableTimestep(self, old_dt):
        from ExchangerLib import exchangeTimestep

        if self.synchronized:
            #print "start new big step"
            mycomm = self.communicator
            self.ccplr_t = exchangeTimestep(old_dt,
                                            mycomm.handle(),
                                            self.sinkComm.handle(),
                                            0)
            self.ecplr_t = 0
            self.synchronized = False

        dt = old_dt
        self.ecplr_t += dt

        # clipping oversized ecplr_t
        if self.ecplr_t >= self.ccplr_t:
            #print "** clip dt **"
            #print "dt = %g, ecplr_t = %g, ccplr_t = %g" %(dt, self.ecplr_t, self.ccplr_t)
            dt = dt - (self.ecplr_t - self.ccplr_t)
            self.ecplr_t = self.ccplr_t
            self.synchronized = True
            #print self.name, " SYNCHRONIZED!"

        # store timestep for interpolating boundary velocities
        self.inlet.storeTimestep(self.ecplr_t, self.ccplr_t)

        #print "%s -   old dt = %g   exchanged dt = %g" % (
        #       self.name, old_dt, dt)
        #print "ccplr_t = %g  ecplr_t = %g" % (self.ccplr_t, self.ecplr_t)
        return dt


    def exchangeSignal(self, signal):
        from ExchangerLib import exchangeSignal
        mycomm = self.communicator
        newsgnl = exchangeSignal(signal,
                                 mycomm.handle(),
                                 self.sinkComm.handle(),
                                 0)
        return newsgnl



    class Inventory(Coupler.Inventory):

        import pyre.inventory as prop

        # excluding nodes in top boundary? (used if vbc is read from file)
        exclude_top = prop.bool("exclude_top", default=False)

        # excluding nodes in bottom boundary?
        exclude_bottom = prop.bool("exclude_bottom", default=False)

        # amending the received vbc to be divergence-free
        amending_outflow = prop.bool("amending_outflow", default=False)



# version
__id__ = "$Id$"

# End of file
