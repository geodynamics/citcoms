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


from pyre.components.Component import Component
import journal

class Controller(Component):


    def __init__(self, name, facility):
        Component.__init__(self, name, facility)

        self.done = False
        self.solver = None
        return


    # Set these attributes as read-only properties, so that they are
    # always in accordance with their counterparts in the C code
    clock = property(lambda self: self.solver.t)
    dt = property(lambda self: self.solver.dt)
    step = property(lambda self: self.solver.step)


    def initialize(self, app):
        self.solver = app.solver
        self.solver.initialize(app)
        return



    def launch(self, app):
        # 0th step
        self.solver.launch(app)

        # do io for 0th step
        self.save()

        ### XXX: if stokes: advection tracers and terminate
        return



    def march(self, totalTime=0, steps=0):
        """explicit time loop"""

        if (self.step + 1) >= steps:
            self.endSimulation()
            return

        while 1:

            # notify solvers we are starting a new timestep
            self.startTimestep()

            # compute an acceptable timestep
            dt = self.stableTimestep()

            # advance
            self.advance(dt)

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


    def startTimestep(self):
        self.solver.newStep()
        return


    def stableTimestep(self):
        dt = self.solver.stableTimestep()
        return dt


    def advance(self, dt):
        self.solver.advance(dt)
        return


    def endTimestep(self, totalTime, steps):
        # are we done?
        if steps and self.step >= steps:
            self.done = True
        if totalTime and self.clock >= totalTime:
            self.done = True

        # solver can terminate time marching by returning True
        self.done = self.solver.endTimestep(self.done)

        return



    def endSimulation(self):
        self.solver.endSimulation()
        return



    def save(self):
        self.solver.save(self.inventory.monitoringFrequency)
        self.solver.checkpoint(self.inventory.checkpointFrequency)
        return



    class Inventory(Component.Inventory):

        import pyre.inventory

        monitoringFrequency = pyre.inventory.int("monitoringFrequency", default=100)
        checkpointFrequency = pyre.inventory.int("checkpointFrequency", default=100)
