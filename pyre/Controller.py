#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.components.SimulationController import SimulationController
import journal

class Controller(SimulationController):

    def __init__(self, name, facility="controller"):
        SimulationController.__init__(self, name, facility)

        self.step = 0
        self.clock = 0.0
        self.solver = None

        return



    def launch(self, app):
        self.solver = app.solver
        self.solver.launch(app)
        self.run_init_simulation()

        # temporary change
        #self.inventory.monitoringFrequency = 1
        return



    def march(self, totalTime=0, steps=0):
        """explicit time loop"""

        while 1:

            # notify solvers we are starting a new timestep
            self.startTimestep()

            # synchronize boundary information
            self.applyBoundaryConditions()

            self.save(self.step)

            # are we done?
            if steps and self.step >= steps:
                break
            if totalTime and self.clock >= totalTime:
                break

            # compute an acceptable timestep
            dt = self.stableTimestep()

            # advance
            self.advance(dt)

            # update smulation clock and step number
            self.clock += dt
            self.step += 1

            # notify solver we finished a timestep
            self.endTimestep()

        # end of time advance loop

        # Notify solver we are done
        self.endSimulation()

        return



    def run_init_simulation(self):
        self.solver.run_init_simulation()
        return



    def endSimulation(self):
        self.solver.endSimulation(self.step)
        return



    def save(self, step):
        if not step % self.inventory.monitoringFrequency:
            self.solver.save(step)
        return


