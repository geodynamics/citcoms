#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.components.Component import Component


class Coupler(Component):


    def __init__(self, name, facility):
        Component.__init__(self, name, facility)

        self.exchanger = None
        self.boundary = None
        return



    def initialize(self, solver):
        # exchanger could be either a FineGridExchanger (FGE)
        # or a CoarseGridExchanger (CGE)
        self.exchanger = solver.inventory.exchanger

        self.exchanger.initialize(solver)
        return



    def launch(self, solver):
        exchanger = self.exchanger
        exchanger.launch()

        # find the common boundary
        print 'exchanging boundary'
        #self.boundary = exchanger.findBoundary()

        # send initial temperature field from CGE to FGE
        #exchanger.initTemperature(self.boundary)
        return



    def applyBoundaryConditions(self):
        return



    def stableTimestep(self, dt):
        return dt




# version
__id__ = "$Id: Coupler.py,v 1.1 2003/09/05 19:49:15 tan2 Exp $"

# End of file
