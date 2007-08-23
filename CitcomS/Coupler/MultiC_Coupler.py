#!/usr/bin/env python

#
#containing coupler with more than one coupled embedded coupler
#

from ContainingCoupler import ContainingCoupler

class MultiC_Coupler(ContainingCoupler):


    def __init__(self, name, facility):
        ContainingCoupler.__init__(self, name, facility)

        self.srcCommList2 = []
        self.sinkComm2 = None
        self.remoteSize2 = 0
        return


    def initialize(self, solver):
        ContainingCoupler.initialize(self, solver)

        self.srcCommList2 = solver.myPlus2

        # number of processors in the remote solver2
        self.remoteSize2 = len(self.srcCommList2)

        # only one of remotePlus2 is sinkComm2
        self.sinkComm2 = solver.remotePlus2[self.communicator.rank]

        # allocate space
        self.remoteBdryList2 = range(self.remoteSize2)
        self.sourceList2 = range(self.remoteSize2)
        self.outletList2 = range(self.remoteSize2)

        return


    def createMesh(self):
        # Create BoundedMesh objects.
        from ExchangerLib import exchangeBoundedBox, createInterior, createEmptyBoundary


        ContainingCoupler.createMesh(self)

        # the bounding box of the mesh on remote solver2
        self.remoteBBox2 = \
                         exchangeBoundedBox(self.globalBBox,
                                            self.communicator.handle(),
                                            self.srcCommList2[0].handle(),
                                            self.srcCommList2[0].size - 1)


        # the nodes within remoteBBox2
        self.interior2, self.myBBox2 = createInterior(self.remoteBBox2,
                                                    self.all_variables)

        # an empty boundary object,\
        # which will be filled by a remote boundary obj.
        for i in range(self.remoteSize2):
            self.remoteBdryList2[i] = createEmptyBoundary()


        return


    def createSource(self):

        ContainingCoupler.createSource(self)

        # the source objects will send boundary conditions to remote sink2
        from ExchangerLib import CitcomSource_create
        for i, comm, b in zip(range(self.remoteSize2),
                              self.srcCommList2,
                              self.remoteBdryList2):
            # sink is always in the last rank of a communicator
            sinkRank2 = comm.size - 1

            # the sources will communicate with the sink in EmbeddedCoupler
            # during creation stage
            self.sourceList2[i] = CitcomSource_create(comm.handle(),
                                                     sinkRank2,
                                                     b,
                                                     self.myBBox2,
                                                     self.all_variables)

        return


    def createSink(self):

        ContainingCoupler.createSink(self)

        # the sink obj. will receive interior
        # temperature from remote sources
        from ExchangerLib import Sink_create

        # the sink will communicate with the source in EmbeddedCoupler
        # during creation stage
        self.sink2 = Sink_create(self.sinkComm2.handle(),
                                self.remoteSize2,
                                self.interior2)
        return


    def createBC(self):

        ContainingCoupler.createBC(self)

        # boundary conditions will be sent by SVTOutlet, which sends
        # stress, velocity, and temperature
        import Outlet
        for i, src in zip(range(self.remoteSize2),
                          self.sourceList2):
            self.outletList2[i] = Outlet.SVTOutlet(src, self.all_variables)
        return

    def createII(self):

        ContainingCoupler.createII(self)


        # interior temperature will be received by TInlet
        import Inlet
        self.inlet2 = Inlet.TInlet(self.interior2,
                                  self.sink2,
                                  self.all_variables)
        return


    # initTemperature

    # restartTemperature

    # modifyT

    def postVSolverRun(self):

        ContainingCoupler.postVSolverRun(self)

        # send computed velocity to ECPLR2 for its BCs
        for outlet in self.outletList2:
            outlet.send()
        return


    def newStep(self):

        ContainingCoupler.newStep(self)

        # update the temperature field in the overlapping region
        if self.inventory.two_way_communication:
            # receive temperture field from EmbeddedCoupler
            self.inlet2.recv()
            self.inlet2.impose()
        return

    def stableTimestep(self, dt):
        #used by controller

        from ExchangerLib import exchangeTimestep
        remote_dt = exchangeTimestep(dt,
                                     self.communicator.handle(),
                                     self.srcCommList[0].handle(),
                                     self.srcCommList[0].size - 1)
        remote_dt2 = exchangeTimestep(dt,
                                     self.communicator.handle(),
                                     self.srcCommList2[0].handle(),
                                     self.srcCommList2[0].size - 1)

        assert remote_dt < dt, \
               'Size of dt in the esolver is greater than dt in the csolver!'
        assert remote_dt2 < dt, \
               'Size of dt in the esolver is greater than dt in the csolver2!'

        #print "%s - old dt = %g   exchanged dt = %g" % (
        #       self.__class__, dt, remote_dt)
        return dt

    def exchangeSignal2(self, signal):
        from ExchangerLib import exchangeSignal
        newsgnl = exchangeSignal(signal,
                                 self.communicator.handle(),
                                 self.srcCommList2[0].handle(),
                                 self.srcCommList2[0].size - 1)
        return newsgnl

    def endTimestep(self, steps, done):

        # exchange predefined signal btwn couplers
        # the signal is used to sync the timesteps
        KEEP_WAITING_SIGNAL = 0
        NEW_STEP_SIGNAL = 1
        END_SIMULATION_SIGNAL = 2
        BIG_NEW_STEP_SIGNAL = 3

        sent = NEW_STEP_SIGNAL

        KEEP_WAITING_FLAG = True
       
        while KEEP_WAITING_FLAG:

            #receive signals
            recv = self.exchangeSignal(sent)
            recv2= self.exchangeSignal2(sent)
            #print "#####"
            #print "recv= %d " % recv, "recv2 = %d" %  recv2
            #print "#####"


            # determining what to send
            if done or (recv == END_SIMULATION_SIGNAL) or \
               (recv2 == END_SIMULATION_SIGNAL):
                # end the simulation    
                sent = END_SIMULATION_SIGNAL
                done = True
                KEEP_WAITING_FLAG = False
            elif (recv == KEEP_WAITING_SIGNAL) or \
                 (recv2 == KEEP_WAITING_SIGNAL):
                sent = NEW_STEP_SIGNAL
            elif (recv == NEW_STEP_SIGNAL) and \
                 (recv2 == NEW_STEP_SIGNAL):
                # tell the embedded couplers to keep going
                sent = BIG_NEW_STEP_SIGNAL
                #print self.name, 'exchanging timestep =', steps
                #print self.name, 'exchanged timestep =', self.coupled_steps
                KEEP_WAITING_FLAG = False
            else:
                raise ValueError, \
                      "Unexpected signal value, singnal = %d" % recv

            # send instructions to embedded couplers
            recv = self.exchangeSignal(sent)
            recv2= self.exchangeSignal2(sent)
            #print "#####"
            #print "sent = %d " % sent
            #print "#####"
            #import sys
            #sys.stdout.flush()

            # this must be put here because it use the same
            # exchangeSignal function. The order of function calls matters.
            if sent == BIG_NEW_STEP_SIGNAL:
                self.coupled_steps = self.exchangeSignal(steps)
                self.coupled_steps2 = self.exchangeSignal2(steps)
 
        return done



# version

__id__ = "$Id$"

# End of file
