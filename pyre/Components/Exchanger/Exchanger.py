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
        self.exchanger = None
        self.catchup = True
        self.done = False
        return



    def selectModule(self):
        import CitcomS.Exchanger
        self.module = CitcomS.Exchanger
        return



    def createExchanger(self, solver):
        raise NotImplementedError
        return



    def findBoundary(self):
        raise NotImplementedError
        return



    def mapBoundary(self):
        raise NotImplementedError
        return



    def initTemperature(self):
        raise NotImplementedError
        return



    def NewStep(self):
        raise NotImplementedError
        return



    def applyBoundaryConditions(self):
        raise NotImplementedError
        return



    def stableTimestep(self, dt):
        raise NotImplementedError
        return



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
            signal = self.module.exchangeSignal(self.exchanger, sent)
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



    class Inventory(Component.Inventory):

        import pyre.properties as prop


        inventory = [

            ]



# version
__id__ = "$Id: Exchanger.py,v 1.9 2003/09/28 20:36:56 tan2 Exp $"

# End of file
