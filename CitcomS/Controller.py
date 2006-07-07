#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#
# CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
# Copyright (C) 2002-2005, California Institute of Technology.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
#</LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


def controller(name="controller", facility="controller"):
    return Controller(name, facility)


from pyre.simulations.SimulationController import SimulationController
import journal

class Controller(SimulationController):


    def __init__(self, name, facility):
        SimulationController.__init__(self, name, facility)

        self.step = 0
        self.clock = 0.0
        self.dt = 0.0
        self.done = False
        self.solver = None
        return



    def initialize(self, app):
        self.solver = app.solver
        self.solver.initialize(app)
        return



    def launch(self, app):
        # 0th step
        self.solver.launch(app)

        # do io for 0th step
        self.save()
        return



    def march(self, totalTime=0, steps=0):
        """explicit time loop"""

        if (self.step + 1) >= steps:
            self.step += 1
            self.endSimulation()
            return

        while 1:

            # notify solvers we are starting a new timestep
            self.startTimestep()

            # synchronize boundary information
            #self.applyBoundaryConditions()

            # compute an acceptable timestep
            self.dt = self.stableTimestep()

            # advance
            self.advance(self.dt)

            # update simulation clock and step number
            from CitcomSLib import return_times
            self.clock, self.dt = return_times(self.solver.all_variables)
            self.step += 1

            # notify solver we finished a timestep
            self.endTimestep(totalTime, steps)

            # do io
            self.save()

            # are we done?
            if self.done:
                break

        # end of time advance loop

        # Notify solver we are done
        self.endSimulation()

        return



    def endTimestep(self, totalTime, steps):
        # are we done?
        if steps and self.step >= steps:
            self.done = True
        if totalTime and self.clock >= totalTime:
            self.done = True

        # solver can terminate time marching by returning True
        self.done = self.solver.endTimestep(self.clock, self.step, self.done)

        return



    def endSimulation(self):
        self.solver.endSimulation(self.step)
        return



    def save(self):
        step = self.step
        self.solver.timesave(self.clock, self.dt, step)
        self.solver.save(step, self.inventory.monitoringFrequency)
        return


