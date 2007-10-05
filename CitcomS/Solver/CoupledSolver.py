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

from CitcomSLib import output, output_time
from Solver import Solver
import journal



class CoupledSolver(Solver):

    def __init__(self, name, facility="solver"):
        Solver.__init__(self, name, facility)

        self.coupler = None
        self.myPlus = []
        self.remotePlus = []
        return


    def initialize(self, application):
        Solver.initialize(self, application)

        self.coupler = application.coupler
        self.myPlus = application.myPlus
        self.remotePlus = application.remotePlus

        self.coupler.initialize(self)
        return


    def launch(self, application):
        #TODO: checkpoint doesn't contain coupler information yet
        self.coupler.launch(self)

        if self.inventory.ic.inventory.restart:
            from CitcomSLib import readCheckpoint
            readCheckpoint(self.all_variables)
        else:
            # initial conditions
            ic = self.inventory.ic
            ic.launch()

            # insure consistent temperature fields across solvers
            self.coupler.exchangeTemperature()

            self.solveVelocities()
        return



    def solveVelocities(self):
        # sync boundary conditions before/after vsolver
        vsolver = self.inventory.vsolver
        self.coupler.preVSolverRun()
        vsolver.run()
        self.coupler.postVSolverRun()
        return



    def advectTracers(self):
        # override Solver.advectTracers, since tracer module
        # doesn't work in coupled run
        return



    def newStep(self):
        Solver.newStep(self)

        # sync the temperature field
        self.coupler.newStep()
        return



    def stableTimestep(self):
        dt = Solver.stableTimestep(self)

        # negotiate with other solver(s)
        dt = self.coupler.stableTimestep(dt)
        return dt


    def endTimestep(self, done):
        done = Solver.endTimestep(self, done)

        # check with other solver, are we done?
        done = self.coupler.endTimestep(self.step, done)
        return done


    def endSimulation(self):
        self._avgCPUTime()
        # write even if not sync'd
        output(self.all_variables, self.step)
        self.finalize()
        return



    def save(self, monitoringFrequency):
        step = self.step

        # for coupled run, output spacing is determined by coupled_steps
        if (not (step % monitoringFrequency)) or (
            not (self.coupler.coupled_steps % monitoringFrequency)):
            output(self.all_variables, step)

        output_time(self.all_variables, step)
        return


    def checkpoint(self, checkpointFrequency):
        Solver.checkpoint(self, checkpointFrequency)

        if not (self.step % checkpointFrequency):
            #TODO: checkpoint for coupler
            pass
        return



# version
__id__ = "$Id$"

# End of file
