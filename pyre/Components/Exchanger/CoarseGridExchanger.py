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


    def createExchanger(self, solver):
        self.exchanger = self.module.createCoarseGridExchanger(
                                     solver.communicator.handle(),
                                     solver.intercomm.handle(),
                                     solver.leader,
                                     solver.localLeader,
                                     solver.remoteLeader,
                                     solver.all_variables
                                     )
        return


    def findBoundary(self):
        # receive boundary from FGE
        self.module.receiveBoundary(self.exchanger)

        # create mapping from boundary to id array
        self.module.mapBoundary(self.exchanger)
        return


    def initTemperature(self):
        self.module.initTemperature(self.exchanger)
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
        self.module.gather(self.exchanger)
        self.module.sendVelocities(self.exchanger)
        return


    def stableTimestep(self, dt):
        new_dt = self.module.exchangeTimestep(self.exchanger, dt)
        #print "%s - old dt = %g   exchanged dt = %g" % (
        #       self.__class__, dt, new_dt)
        return min(dt, new_dt)



    class Inventory(Exchanger.Inventory):

        import pyre.properties as prop


        inventory = [

            ]



# version
__id__ = "$Id: CoarseGridExchanger.py,v 1.17 2003/10/01 22:04:41 tan2 Exp $"

# End of file
