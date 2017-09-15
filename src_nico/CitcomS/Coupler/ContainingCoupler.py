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

class ContainingCoupler(Coupler):


    def __init__(self, name, facility):
        Coupler.__init__(self, name, facility)
        return


    def initialize(self, solver):
        Coupler.initialize(self, solver)
        return


    def createMesh(self):
        # Create BoundedMesh objects.
     
        from ExchangerLib import createGlobalBoundedBox, exchangeBoundedBox, createInterior, createEmptyBoundary

        # the bounding box of the mesh on this solver
        self.globalBBox = createGlobalBoundedBox(self.all_variables)

        # the bounding box of the mesh on the other solver
        self.remoteBBox = exchangeBoundedBox(self.globalBBox,
                                             self.communicator.handle(),
                                             self.srcCommList[0].handle(),
                                             self.srcCommList[0].size - 1)

        # the nodes within remoteBBox
        self.interior, self.myBBox = createInterior(self.remoteBBox,
                                                    self.all_variables)

        self.remoteBdryList = range(self.remoteSize)
        for i in range(self.remoteSize):
            # an empty boundary object for remote velocity nodes
            # will be filled by a remote boundary obj.
            self.remoteBdryList[i] = createEmptyBoundary()

        return


    def createSourceSink(self):
        # create source first, then sink. The order is important.
        self.createSource()

        if self.inventory.two_way_communication:
            self.createSink()
        return


    def createSource(self):
        # the source obj's will send boundary conditions to a remote sink
        from ExchangerLib import CitcomSource_create
        self.sourceList = range(self.remoteSize)
        for i, comm, b in zip(range(self.remoteSize),
                              self.srcCommList,
                              self.remoteBdryList):
            # sink is always in the last rank of a communicator
            sinkRank = comm.size - 1

            # the sources will communicate with the sink in EmbeddedCoupler
            # during creation stage
            self.sourceList[i] = CitcomSource_create(comm.handle(),
                                                     sinkRank,
                                                     b,
                                                     self.myBBox,
                                                     self.all_variables)

        if self.inventory.exchange_pressure:
            from ExchangerLib import createEmptyPInterior
            self.pinterior = range(self.remoteSize)

            for i in range(self.remoteSize):
                self.pinterior[i] = createEmptyPInterior()

            self.psourceList = range(self.remoteSize)
            for i, comm, b in zip(range(self.remoteSize),
                                  self.srcCommList,
                                  self.pinterior):
                # sink is always in the last rank of a communicator
                sinkRank = comm.size - 1
                self.psourceList[i] = CitcomSource_create(comm.handle(),
                                                          sinkRank,
                                                          b,
                                                          self.myBBox,
                                                          self.all_variables)
        return


    def createSink(self):
        # the sink obj. will receive interior temperature from remote sources
        from ExchangerLib import Sink_create

        # the sink will communicate with the source in EmbeddedCoupler
        # during creation stage
        self.sink = Sink_create(self.sinkComm.handle(),
                                self.remoteSize,
                                self.interior)
        return


    def createBC(self):
        # boundary conditions will be sent by SVTOutlet, which sends
        # stress, velocity, and temperature
        import Outlet
        self.outletList = range(self.remoteSize)
        for i, src in zip(range(self.remoteSize),
                          self.sourceList):
            self.outletList[i] = Outlet.SVTOutlet(src, self.all_variables)

        if self.inventory.exchange_pressure:
            self.poutletList = range(self.remoteSize)
            for i, src in zip(range(self.remoteSize),
                              self.psourceList):

                self.poutletList[i] = Outlet.POutlet(src, self.all_variables)

        return


    def createII(self):
        # interior temperature will be received by TInlet
        import Inlet
        self.inlet = Inlet.TInlet(self.interior,
                                  self.sink,
                                  self.all_variables)
        return


    def initTemperature(self):
        from ExchangerLib import initTemperature
        initTemperature(self.remoteBBox,
                        self.all_variables)
        return


    def exchangeTemperature(self):
        if not self.inventory.exchange_initial_temperature:
            return

        from ExchangerLib import createEmptyInterior, CitcomSource_create
        interior = range(self.remoteSize)
        source = range(self.remoteSize)

        for i in range(len(interior)):
            interior[i] = createEmptyInterior()

        for i, comm, b in zip(range(self.remoteSize),
                              self.srcCommList,
                              interior):
            # sink is always in the last rank of a communicator
            sinkRank = comm.size - 1
            source[i] = CitcomSource_create(comm.handle(),
                                            sinkRank,
                                            b,
                                            self.myBBox,
                                            self.all_variables)

        import Outlet
        for i, src in zip(range(self.remoteSize), source):
            outlet = Outlet.TOutlet(src, self.all_variables)
            outlet.send()

        # Any modification of read-in temperature is done here
        # Note: modifyT is called after sending unmodified T to EmbeddedCoupler.
        # If T is modified before sending, EmbeddedCoupler's T will lose sharp
        # feature.
        # EmbeddedCoupler has to call modifyT too to ensure consistent T field.
        #self.modifyT(self.remoteBBox)

        return


    def postVSolverRun(self):
        # send computed velocity to ECPLR for its BCs
        for outlet in self.outletList:
            outlet.send()

        if self.inventory.exchange_pressure:
            for outlet in self.poutletList:
                outlet.send()
        return


    def newStep(self):
        # update the temperature field in the overlapping region
        if self.inventory.two_way_communication:
            # receive temperture field from EmbeddedCoupler
            self.inlet.recv()
            self.inlet.impose()
        return


    def stableTimestep(self, dt):
        from ExchangerLib import exchangeTimestep
        remote_dt = exchangeTimestep(dt,
                                     self.communicator.handle(),
                                     self.srcCommList[0].handle(),
                                     self.srcCommList[0].size - 1)

        assert remote_dt < dt, \
               'Size of dt in the esolver is greater than dt in the csolver!'

        #print "%s - old dt = %g   exchanged dt = %g" % (
        #       self.__class__, dt, remote_dt)
        return dt


    def exchangeSignal(self, signal):
        from ExchangerLib import exchangeSignal
        newsgnl = exchangeSignal(signal,
                                 self.communicator.handle(),
                                 self.srcCommList[0].handle(),
                                 self.srcCommList[0].size - 1)
        return newsgnl



    class Inventory(Coupler.Inventory):

        import pyre.inventory as prop






# version
__id__ = "$Id: ContainingCoupler.py 15108 2009-06-02 22:56:46Z tan2 $"

# End of file
