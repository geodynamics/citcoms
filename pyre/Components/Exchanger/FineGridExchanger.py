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


    def createExchanger(self, solver):
        self.exchanger = self.module.createFineGridExchanger(
                                     solver.communicator.handle(),
                                     solver.intercomm.handle(),
                                     solver.localLeader,
                                     solver.remoteLeader,
                                     solver.all_variables
                                     )
        return



    def findBoundary(self):
        boundary = self.module.createBoundary(self.exchanger)

        # send boundary from CGE
        self.module.sendBoundary(self.exchanger, boundary)

	# create mapping from boundary to id array
	self.module.mapBoundary(self.exchanger, boundary)
        return boundary



    def initTemperature(self):
        # receive temperture field from CGE
        self.module.receiveTemperature(self.exchanger)
        return



    def waitNewStep(self):
        # no wait
        return



    def applyBoundaryConditions(self):
        self.module.gather(self.exchanger)
        self.module.send(self.exchanger)
        return


    class Inventory(Exchanger.Inventory):

        import pyre.properties as prop


        inventory = [

            ]



# version
__id__ = "$Id: FineGridExchanger.py,v 1.3 2003/09/09 21:04:45 tan2 Exp $"

# End of file
