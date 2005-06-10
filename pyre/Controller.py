#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#=====================================================================
#
#                             CitcomS.py
#                 ---------------------------------
#
#                              Authors:
#            Eh Tan, Eun-seo Choi, and Pururav Thoutireddy 
#          (c) California Institute of Technology 2002-2005
#
#        By downloading and/or installing this software you have
#       agreed to the CitcomS.py-LICENSE bundled with this software.
#             Free for non-commercial academic research ONLY.
#      This program is distributed WITHOUT ANY WARRANTY whatsoever.
#
#=====================================================================
#
#  Copyright June 2005, by the California Institute of Technology.
#  ALL RIGHTS RESERVED. United States Government Sponsorship Acknowledged.
# 
#  Any commercial use must be negotiated with the Office of Technology
#  Transfer at the California Institute of Technology. This software
#  may be subject to U.S. export control laws and regulations. By
#  accepting this software, the user agrees to comply with all
#  applicable U.S. export laws and regulations, including the
#  International Traffic and Arms Regulations, 22 C.F.R. 120-130 and
#  the Export Administration Regulations, 15 C.F.R. 730-744. User has
#  the responsibility to obtain export licenses, or other export
#  authority as may be required before exporting such information to
#  foreign countries or providing access to foreign nationals.  In no
#  event shall the California Institute of Technology be liable to any
#  party for direct, indirect, special, incidental or consequential
#  damages, including lost profits, arising out of the use of this
#  software and its documentation, even if the California Institute of
#  Technology has been advised of the possibility of such damage.
# 
#  The California Institute of Technology specifically disclaims any
#  warranties, including the implied warranties or merchantability and
#  fitness for a particular purpose. The software and documentation
#  provided hereunder is on an "as is" basis, and the California
#  Institute of Technology has no obligations to provide maintenance,
#  support, updates, enhancements or modifications.
#
#=====================================================================
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
            dt = self.stableTimestep()

            # advance
            self.advance(dt)

            # update smulation clock and step number
            self.clock += dt
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
        self.solver.timesave(self.clock, step)
        self.solver.save(step, self.inventory.monitoringFrequency)
        return


