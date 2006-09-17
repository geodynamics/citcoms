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

from CitcomSLib import CPU_time, output, output_time, return_dt, return_t, return_step
from pyre.components.Component import Component
import journal



class Solver(Component):

    def __init__(self, name, facility="solver"):
        Component.__init__(self, name, facility)

        self.all_variables = None
        self.communicator = None
	self.start_cpu_time = 0
        return


    def _dt(self):
        '''get the value of dt from the C code'''
        return return_dt(self.all_variables)


    def _t(self):
        '''get the value of t from the C code'''
        return return_t(self.all_variables)


    def _step(self):
        '''get the value of step from the C code'''
        return return_step(self.all_variables)


    # Set these attributes as read-only properties, so that they are
    # always in accordance with their counterparts in the C code
    t = property(_t)
    dt = property(_dt)
    step = property(_step)


    def initialize(self, application):

        from CitcomSLib import citcom_init, global_default_values, set_signal

        comm = application.solverCommunicator
        all_variables = citcom_init(comm.handle())
	self.communicator = comm
        self.all_variables = all_variables

        self.initializeSolver()

        # information about clock time
	self.start_cpu_time = CPU_time()

	inv = self.inventory

        inv.mesher.initialize(all_variables)
        inv.tsolver.initialize(all_variables)
        inv.vsolver.initialize(all_variables)

        inv.bc.initialize(all_variables)
        inv.const.initialize(all_variables)
        inv.ic.initialize(all_variables)
        inv.param.initialize(all_variables)
        inv.phase.initialize(all_variables)
        inv.tracer.initialize(all_variables)
        inv.visc.initialize(all_variables)

        global_default_values(self.all_variables)
        set_signal()

        self.setProperties()

        self.restart = self.inventory.ic.inventory.restart

        self.ic_initTemperature = self.inventory.ic.initTemperature
        return


    def launch(self, application):
        self._setup()

        # initial conditions
        ic = self.inventory.ic
        ic.launch()

        self.solveVelocities()
        return


    def _setup(self):
        mesher = self.inventory.mesher
        mesher.setup()

        vsolver = self.inventory.vsolver
        vsolver.setup()

        tsolver = self.inventory.tsolver
        tsolver.setup()

        # create mesh
        mesher.run()

        # initialize const. related to mesh
        vsolver.launch()
        tsolver.launch()
        return


    def solveVelocities(self):
        vsolver = self.inventory.vsolver
        vsolver.run()
        return



    def solveTemperature(self, dt):
        tsolver = self.inventory.tsolver
        tsolver.run(dt)
        return



    def solveAdditional(self):
        self.inventory.tracer.run()
        return



    def newStep(self):
        return



    def stableTimestep(self):
        tsolver = self.inventory.tsolver
        dt = tsolver.stable_timestep()
        return dt



    def advance(self, dt):
        self.solveTemperature(dt)
        self.solveVelocities()
        self.solveAdditional()
        return


    def endTimestep(self, done):
        self.inventory.visc.updateMaterial()
        self.inventory.bc.updatePlateVelocity()
        return done


    def endSimulation(self):
        self._avgCPUTime()
        self.finalize()
        return


    def _avgCPUTime(self):
        step = self.step
        total_cpu_time = CPU_time() - self.start_cpu_time

        rank = self.communicator.rank
        if not rank:
            import sys
            sys.stderr.write("Average cpu time taken for velocity step = %f\n"
                             % (total_cpu_time / (step+1)) )
        return


    def save(self, monitoringFrequency):
        step = self.step

        # output spacing is 'monitoringFrequency'
        if not (step % monitoringFrequency):
            output(self.all_variables, step)

        output_time(self.all_variables, step)
        return


    def setProperties(self):

        from CitcomSLib import Solver_set_properties

        Solver_set_properties(self.all_variables, self.inventory)

	inv = self.inventory
        inv.mesher.setProperties()
        inv.tsolver.setProperties()
        inv.vsolver.setProperties()

        inv.bc.setProperties()
        inv.const.setProperties()
        inv.ic.setProperties()
        inv.param.setProperties()
        inv.phase.setProperties()
        inv.tracer.setProperties()
        inv.visc.setProperties()

        return


    def finalize(self):
        from CitcomSLib import output_finalize
        output_finalize(self.all_variables)
        return


    class Inventory(Component.Inventory):

        import pyre.inventory

        # component modules
        import CitcomS.Components.Advection_diffusion as Advection_diffusion
        import CitcomS.Components.Stokes_solver as Stokes_solver

        # components
        from CitcomS.Components.BC import BC
        from CitcomS.Components.Const import Const
        from CitcomS.Components.IC import IC
        from CitcomS.Components.Param import Param
        from CitcomS.Components.Phase import Phase
        from CitcomS.Components.Tracer import Tracer
        from CitcomS.Components.Visc import Visc


        tsolver = pyre.inventory.facility("tsolver", factory=Advection_diffusion.temperature_diffadv)
        vsolver = pyre.inventory.facility("vsolver", factory=Stokes_solver.incompressibleNewtonian)

        bc = pyre.inventory.facility("bc", factory=BC)
        const = pyre.inventory.facility("const", factory=Const)
        ic = pyre.inventory.facility("ic", factory=IC)
        param = pyre.inventory.facility("param", factory=Param)
        phase = pyre.inventory.facility("phase", factory=Phase)
        tracer = pyre.inventory.facility("tracer", factory=Tracer)
        visc = pyre.inventory.facility("visc", factory=Visc)

        datadir = pyre.inventory.str("datadir", default=".")

        rayleigh = pyre.inventory.float("rayleigh", default=1e+05)
        Q0 = pyre.inventory.float("Q0", default=0.0)

        stokes_flow_only = pyre.inventory.bool("stokes_flow_only", default=False)

        output_format = pyre.inventory.str("output_format", default="ascii",
                            validator=pyre.inventory.choice(["ascii", "hdf5"]))
        output_optional = pyre.inventory.str("output_optional", default="")

        verbose = pyre.inventory.bool("verbose", default=False)
        see_convergence = pyre.inventory.bool("see_convergence", default=True)


# version
__id__ = "$Id$"

# End of file
