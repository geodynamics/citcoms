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
        # send temperture field to FGE
        #self.module.sendTemperature(self.exchanger)
        return


    def solveVelocities(self, vsolver):
        vsolver.run()
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
        old_dt = dt
        self.module.exchangeTimestep(self.exchanger, dt)
        #print "%s - old dt = %g   exchanged dt = %g" % (self.__class__, old_dt, dt)
        return dt



    class Inventory(Exchanger.Inventory):

        import pyre.properties as prop


        inventory = [

            ]



# version
__id__ = "$Id: CoarseGridExchanger.py,v 1.16 2003/09/30 01:50:24 tan2 Exp $"

# End of file
