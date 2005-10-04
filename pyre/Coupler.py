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
__id__ = "$Id$"

# End of file
