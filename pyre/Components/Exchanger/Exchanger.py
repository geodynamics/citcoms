#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.components.Component import Component


class Exchanger(Component):


    def __init__(self, name, facility):
        Component.__init__(self, name, facility)

        self.module = None
        self.mesh = None
        self.all_variables = None
        self.communicator = None
        self.srcComm = []
        self.sinkComm = None
        self.numSrc = 0

        self.sink = {}
        self.source = {}

        self.catchup = True
        self.done = False
        return


    def initialize(self, solver):
        self.selectModule()
        self.all_variables = solver.all_variables
        self.communicator = solver.communicator
        self.srcComm = solver.myPlus
        self.numSrc = len(self.srcComm)

        # only one of remotePlus is sinkComm
        self.sinkComm = solver.remotePlus[self.communicator.rank]
        return



    def launch(self, solver):
        self.createMesh()
        self.createSourceSink()
        self.createBC()
        self.createII()
        return



    def selectModule(self):
        import CitcomS.Exchanger
        self.module = CitcomS.Exchanger
        return



    def createMesh(self):
        raise NotImplementedError
        return



    def createSourceSink(self):
        raise NotImplementedError
        return



    def createBC(self):
        raise NotImplementedError
        return



    def initTemperature(self):
        raise NotImplementedError
        return



    def preVSolverRun(self):
        # do nothing, overridden by FGE
        return



    def postVSolverRun(self):
        # do nothing, overridden by CGE
        return



    def NewStep(self):
        raise NotImplementedError
        return



    def applyBoundaryConditions(self):
        raise NotImplementedError
        return



    def stableTimestep(self, dt):
        raise NotImplementedError
        return dt



    def endTimestep(self, done):
        KEEP_WAITING_SIGNAL = 0
        NEW_STEP_SIGNAL = 1
        END_SIMULATION_SIGNAL = 2

        if done:
            sent = END_SIMULATION_SIGNAL
        elif self.catchup:
            sent = NEW_STEP_SIGNAL
        else:
            sent = KEEP_WAITING_SIGNAL
        #print "    ", self.name, "  send: ", sent

        while 1:
            signal = self.exchangeSignal(sent)

            if sent == END_SIMULATION_SIGNAL:
                signal = END_SIMULATION_SIGNAL
            #print "    ", self.name, " received  ", signal

            if signal == KEEP_WAITING_SIGNAL:
                pass
            elif signal == NEW_STEP_SIGNAL:
                break
            elif signal == END_SIMULATION_SIGNAL:
                done = True
                break
            else:
                raise ValueError, "Unexpected signal value, singnal = %d" % signal

        return done



    def exchangeSignal(self, signal):
        raise NotImplementedError
        return signal




    class Inventory(Component.Inventory):

        import pyre.properties as prop


        inventory = [

            ]



# version
__id__ = "$Id: Exchanger.py,v 1.12 2003/11/11 19:29:50 tan2 Exp $"

# End of file
