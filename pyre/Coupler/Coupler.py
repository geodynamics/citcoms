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

        # choose c++ exchanger module
        self.exchanger.selectModule()
        # create c++ exchanger
        self.exchanger.createExchanger(solver)
        return



    def launch(self, solver):
        self.boundary = self.exchanger.findBoundary()
        return



    def initTemperature(self):
        # send initial temperature field from CGE to FGE
        self.exchanger.initTemperature()
        return



    def solveVelocities(self, vsolver):
        self.exchanger.solveVelocities(vsolver)
        return



    def newStep(self):
        #self.exchanger.NewStep()
        return



    def applyBoundaryConditions(self):
        self.exchanger.applyBoundaryConditions()
        return



    def stableTimestep(self, dt):
        dt = self.exchanger.stableTimestep(dt)
        return dt



    def endTimestep(self, done):
        done = self.exchanger.endTimestep(done)
        return done


# version
__id__ = "$Id: Coupler.py,v 1.6 2003/09/30 01:48:34 tan2 Exp $"

# End of file
