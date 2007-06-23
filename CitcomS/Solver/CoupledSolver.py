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
        print self.name, 'enter initialize'
        Solver.initialize(self, application)
        print self.name, self.all_variables

        self.coupler = application.coupler
        self.myPlus = application.myPlus
        self.remotePlus = application.remotePlus

        self.restart = self.inventory.ic.inventory.restart
        self.ic_initTemperature = self.inventory.ic.initTemperature

        self.coupler.initialize(self)
        print self.name, 'exit initialize'
        return


    def launch(self, application):
        print self.name, 'enter launch'
        self._setup()

        self.coupler.launch(self)

        ic = self.inventory.ic
        if not (ic.inventory.restart or ic.inventory.post_p):
            # switch the default initTemperature to coupled version
            ic.initTemperature = self.coupler.initTemperature

        # initial conditions
        ic.launch()

        self.solveVelocities()
        print self.name, 'exit launch'
        return



    def solveVelocities(self):
        vsolver = self.inventory.vsolver
        self.coupler.preVSolverRun()
        vsolver.run()
        self.coupler.postVSolverRun()
        return



    def solveAdditional(self):
        # override Solver.solveAdditional, since tracer module
        # doesn't work in coupled run
        return



    def newStep(self):
        Solver.newStep(self)
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





# version
__id__ = "$Id$"

# End of file
