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
        return


    def createExchanger(self, solver):
        self.exchanger = self.module.createFineGridExchanger(
                                     solver.communicator.handle(),
                                     solver.intercomm.handle(),
                                     solver.leader,
                                     solver.localLeader,
                                     solver.remoteLeader,
                                     solver.all_variables
                                     )

        return


    def findBoundary(self):
        self.module.createBoundary(self.exchanger)

        # send boundary from CGE
        self.module.sendBoundary(self.exchanger)

	# create mapping from boundary node # to global node #
	self.module.mapBoundary(self.exchanger)
        return


    def initTemperature(self):
        # receive temperture field from CGE
        #self.module.receiveTemperature(self.exchanger)
        return


    def solveVelocities(self, vsolver):
        if self.catchup:
            self.applyBoundaryConditions()

        vsolver.run()
        return


    def NewStep(self):
        #if self.catchup:
            # send temperture field to CGE
            #self.module.sendTemperature(self.exchanger)

        return


    def applyBoundaryConditions(self):
        self.module.receiveVelocities(self.exchanger)
        self.module.distribute(self.exchanger)
        self.module.imposeBC(self.exchanger)
        return


    def stableTimestep(self, dt):
        if self.catchup:
            self.cge_t = self.module.exchangeTimestep(self.exchanger, dt)
            self.fge_t = 0
            self.catchup = False

        self.fge_t += dt
        old_dt = dt

        if self.fge_t >= self.cge_t:
            dt = dt - (self.fge_t - self.cge_t)
            self.fge_t = self.cge_t
            self.catchup = True

        # store timestep for interpolating boundary velocities
        self.module.storeTimestep(self.exchanger, self.fge_t, self.cge_t)
        #print "%s - old dt = %g   exchanged dt = %g" % (self.__class__, old_dt, dt)
        return dt



    class Inventory(Exchanger.Inventory):

        import pyre.properties as prop


        inventory = [

            ]



# version
__id__ = "$Id: FineGridExchanger.py,v 1.17 2003/09/30 01:50:24 tan2 Exp $"

# End of file
