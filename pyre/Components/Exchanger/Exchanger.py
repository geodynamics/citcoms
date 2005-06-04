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
        self.coupled_steps = 1
        return


    def initialize(self, solver):
        self.selectModule()
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

        if self.inventory.two_way_communication:
            self.createII()
        return


    def selectModule(self):
        import CitcomS.Exchanger
        self.module = CitcomS.Exchanger
        return


    def modifyT(self, bbox):
        self.module.modifyT(bbox, self.all_variables)
        return


    def preVSolverRun(self):
        # do nothing, overridden by FGE
        return


    def postVSolverRun(self):
        # do nothing, overridden by CGE
        return


    def endTimestep(self, steps, done):
        KEEP_WAITING_SIGNAL = 0
        NEW_STEP_SIGNAL = 1
        END_SIMULATION_SIGNAL = 2

        if done:
            sent = END_SIMULATION_SIGNAL
        elif self.catchup:
            sent = NEW_STEP_SIGNAL
        else:
            sent = KEEP_WAITING_SIGNAL

        while 1:
            signal = self.exchangeSignal(sent)

            if done or (signal == END_SIMULATION_SIGNAL):
                done = True
                break
            elif signal == KEEP_WAITING_SIGNAL:
                pass
            elif signal == NEW_STEP_SIGNAL:
                if self.catchup:
                    #print self.name, 'exchanging timestep =', steps
                    self.coupled_steps = self.exchangeSignal(steps)
                    #print self.name, 'exchanged timestep =', self.coupled_steps
                break
            else:
                raise ValueError, \
                      "Unexpected signal value, singnal = %d" % signal

        return done



    class Inventory(Component.Inventory):

        import pyre.inventory as prop



        two_way_communication = prop.bool("two_way_communication", default=True)

        # if dimensional is True, quantities exchanged are dimensional
        dimensional = prop.bool("dimensional", default=True)
        # if transformational is True, quantities exchanged are in standard coordiate system
        transformational = prop.bool("transformational", default=True)




# version
__id__ = "$Id: Exchanger.py,v 1.22 2005/06/03 21:51:44 leif Exp $"

# End of file
