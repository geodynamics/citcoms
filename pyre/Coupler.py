#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


def coupler(name="coupler", facility="coupler"):
    return Coupler(name, facility)


from pyre.components.Component import Component


class Coupler(Component):


    def __init__(self, name, facility):
        Component.__init__(self, name, facility)

        self.exchanger = None
        return



    def initialize(self, solver):
        # exchanger could be either a FineGridExchanger (FGE)
        # or a CoarseGridExchanger (CGE)
        self.exchanger = solver.exchanger
        self.exchanger.initialize(solver)
        return



    def launch(self, solver):
        self.exchanger.launch(solver)
        return



    def initTemperature(self):
        # send initial temperature field from CGE to FGE
        self.exchanger.initTemperature()
        return



    def preVSolverRun(self):
        self.exchanger.preVSolverRun()
        return



    def postVSolverRun(self):
        self.exchanger.postVSolverRun()
        return



    def newStep(self):
        self.exchanger.NewStep()
        return



    def applyBoundaryConditions(self):
        self.exchanger.applyBoundaryConditions()
        return



    def stableTimestep(self, dt):
        dt = self.exchanger.stableTimestep(dt)
        return dt



    def endTimestep(self, steps, done):
        done = self.exchanger.endTimestep(steps, done)
        return done


# version
__id__ = "$Id: Coupler.py,v 1.11 2004/08/07 22:08:35 tan2 Exp $"

# End of file
