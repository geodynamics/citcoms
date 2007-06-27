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

        # exchanged information is non-dimensional
        self.inventory.dimensional = False
        # exchanged information is in spherical coordinate
        self.inventory.transformational = False

        return


    def initialize(self, solver):
        Coupler.initialize(self, solver)

	# restart and use temperautre field of previous run?
        self.restart = solver.restart
        if self.restart:
            self.ic_initTemperature = solver.ic_initTemperature

	self.all_variables = solver.all_variables

        # allocate space for exchanger objects
        self.boundary = range(self.numSrc)
        self.source["BC"] = range(self.numSrc)
        self.BC = range(self.numSrc)

        # init'd Convertor singleton, this must be done before any other
        # exchanger call
        from ExchangerLib import initConvertor
        initConvertor(self.inventory.dimensional,
                      self.inventory.transformational,
                      self.all_variables)

        return


    def createMesh(self):
        '''Create BoundedMesh objects.
        '''
        from ExchangerLib import createGlobalBoundedBox, exchangeBoundedBox, createInterior, createEmptyBoundary

        # the bounding box of the mesh on this solver
        self.globalBBox = createGlobalBoundedBox(self.all_variables)

        # the bounding box of the mesh on the other solver
        self.remoteBBox = exchangeBoundedBox(self.globalBBox,
                                             self.communicator.handle(),
                                             self.srcComm[0].handle(),
                                             self.srcComm[0].size - 1)

        # the nodes within remoteBBox
        self.interior, self.myBBox = createInterior(self.remoteBBox,
                                                    self.all_variables)

        # an empty boundary object, which will be filled by a remote boundary obj.
        for i in range(len(self.boundary)):
            self.boundary[i] = createEmptyBoundary()

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
        for i, comm, b in zip(range(self.numSrc),
                              self.srcComm,
                              self.boundary):
            # sink is always in the last rank of a communicator
            sinkRank = comm.size - 1
            self.source["BC"][i] = CitcomSource_create(comm.handle(),
                                                       sinkRank,
                                                       b,
                                                       self.myBBox,
                                                       self.all_variables)

        return


    def createSink(self):
        # the sink obj. will receive interior temperature from remote sources
        from ExchangerLib import Sink_create
        self.sink["Intr"] = Sink_create(self.sinkComm.handle(),
                                        self.numSrc,
                                        self.interior)
        return


    def createBC(self):
        # boundary conditions will be sent by SVTOutlet, which sends
        # stress, velocity, and temperature
        import Outlet
        for i, src in zip(range(self.numSrc),
                          self.source["BC"]):
            self.BC[i] = Outlet.SVTOutlet(src,
                                          self.all_variables)
        return


    def createII(self):
        # interior temperature will be received by TInlet
        import Inlet
        self.II = Inlet.TInlet(self.interior,
                               self.sink["Intr"],
                               self.all_variables)
        return


    def initTemperature(self):
        if self.restart:
            # read-in restarted temperature field
            self.ic_initTemperature()
            del self.ic_initTemperature
            # send temperature to EmbeddedCoupler and postprocess
            self.restartTemperature()
        else:
            from ExchangerLib import initTemperature
            initTemperature(self.remoteBBox,
                            self.all_variables)
        return


    def restartTemperature(self):
        from ExchangerLib import createEmptyInterior, CitcomSource_create
        interior = range(self.numSrc)
        source = range(self.numSrc)

        for i in range(len(interior)):
            interior[i] = createEmptyInterior()

        for i, comm, b in zip(range(self.numSrc),
                              self.srcComm,
                              interior):
            # sink is always in the last rank of a communicator
            sinkRank = comm.size - 1
            source[i] = CitcomSource_create(comm.handle(),
                                            sinkRank,
                                            b,
                                            self.myBBox,
                                            self.all_variables)

        import Outlet
        for i, src in zip(range(self.numSrc), source):
            outlet = Outlet.TOutlet(src, self.all_variables)
            outlet.send()

        # Any modification of read-in temperature is done here
        # Note: modifyT is called after sending unmodified T to EmbeddedCoupler.
        # If T is modified before sending, EmbeddedCoupler's T will lose sharp
        # feature.
        # EmbeddedCoupler has to call modifyT too to ensure consistent T field.
        self.modifyT(self.remoteBBox)

        return


    def postVSolverRun(self):
        # send computed velocity to ECPLR for its BCs
        self.applyBoundaryConditions()
        return


    def newStep(self):
        # update the temperature field in the overlapping region
        if self.inventory.two_way_communication:
            # receive temperture field from EmbeddedCoupler
            self.II.recv()
            self.II.impose()
        return


    def applyBoundaryConditions(self):
        for bc in self.BC:
            bc.send()
        return


    def stableTimestep(self, dt):
        from ExchangerLib import exchangeTimestep
        new_dt = exchangeTimestep(dt,
                                  self.communicator.handle(),
                                  self.srcComm[0].handle(),
                                  self.srcComm[0].size - 1)
        #print "%s - old dt = %g   exchanged dt = %g" % (
        #       self.__class__, dt, new_dt)
        return dt


    def exchangeSignal(self, signal):
        from ExchangerLib import exchangeSignal
        newsgnl = exchangeSignal(signal,
                                 self.communicator.handle(),
                                 self.srcComm[0].handle(),
                                 self.srcComm[0].size - 1)
        return newsgnl



    class Inventory(Coupler.Inventory):

        import pyre.inventory as prop






# version
__id__ = "$Id$"

# End of file
