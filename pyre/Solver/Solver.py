#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.components.Solver import Solver as BaseSolver
import journal


class Solver(BaseSolver):


    def launch(self,application):
	#journal.info("staging").log("setup MPI")
        comm = application.solverCommunicator
        self.all_variables = self.CitcomModule.citcom_init(comm.handle())

        self.initialize()
	self.CitcomModule.global_default_values(self.all_variables)
        self.CitcomModule.set_signal()
        self.setProperties()

	self._start_cpu_time = self.CitcomModule.CPU_time()

	self.rank = comm.rank
	print "my rank is ", self.rank
        return



    def run_init_simulation(self):
        mesher = self.inventory.mesher
        mesher.setup()

        vsolver = self.inventory.vsolver
        vsolver.setup()

        tsolver = self.inventory.tsolver
        tsolver.setup()

        mesher.run()
        vsolver.run()
        tsolver.launch()

        return



    def advance(self,dt):
        self._loopInfo.log(
            "%s: step %d: advancing the solution by dt = %s" % (self.name, self.step, dt))

        vsolver = self.inventory.vsolver
        vsolver.setup()

        tsolver = self.inventory.tsolver
        tsolver.setup()

        tsolver.run(dt)
        vsolver.run()

        return



    def stableTimestep(self):
        tsolver = self.inventory.tsolver
        dt=tsolver.stable_timestep()
        self._loopInfo.log(
            "%s: step %d: stable timestep dt = %s" % (self.name, self.step, dt))
        return dt



    def endSimulation(self, step):
        total_cpu_time = self.CitcomModule.CPU_time() - self._start_cpu_time
        if not self.rank:
            print "Average cpu time taken for velocity step = %f" % (
                total_cpu_time / step )

	#self.CitcomModule.finalize()

        return



    def save(self,step):
        self.CitcomModule.output(self.all_variables,step)
        return



    def initialize(self):
	inv = self.inventory
        CitcomModule = self.CitcomModule
        all_variables = self.all_variables

        inv.mesher.initialize(CitcomModule, all_variables)
        inv.tsolver.initialize(CitcomModule, all_variables)
        inv.vsolver.initialize(CitcomModule, all_variables)

        inv.bc.initialize(CitcomModule, all_variables)
        inv.const.initialize(CitcomModule, all_variables)
        inv.ic.initialize(CitcomModule, all_variables)
        inv.param.initialize(CitcomModule, all_variables)
        inv.phase.initialize(CitcomModule, all_variables)
        inv.visc.initialize(CitcomModule, all_variables)
        return



    def setProperties(self):
	inv = self.inventory

        inv.mesher.setProperties()
        inv.tsolver.setProperties()
        inv.vsolver.setProperties()

        inv.bc.setProperties()
        inv.const.setProperties()
        inv.ic.setProperties()
        inv.param.setProperties()
        inv.phase.setProperties()
        inv.visc.setProperties()

        return



    class Inventory(BaseSolver.Inventory):

        import pyre.facilities

        # facilities
        from CitcomS.Facilities.TSolver import TSolver
        from CitcomS.Facilities.VSolver import VSolver

        # component modules
        import CitcomS.Components.Advection_diffusion as Advection_diffusion
        import CitcomS.Components.Stokes_solver as Stokes_solver

        # components
        from CitcomS.Components.BC import BC
        from CitcomS.Components.Const import Const
        from CitcomS.Components.IC import IC
        from CitcomS.Components.Param import Param
        from CitcomS.Components.Phase import Phase
        from CitcomS.Components.Visc import Visc

        inventory = [

            TSolver("tsolver", Advection_diffusion.temperature_diffadv("temp")),
            VSolver("vsolver", Stokes_solver.incompressibleNewtonian("incomp-newtonian")),

            pyre.facilities.facility("bc", default=BC("bc", "bc")),
            pyre.facilities.facility("const", default=Const("const", "const")),
            pyre.facilities.facility("ic", default=IC("ic", "ic")),
            pyre.facilities.facility("param", default=Param("param", "param")),
            pyre.facilities.facility("phase", default=Phase("phase", "phase")),
            pyre.facilities.facility("visc", default=Visc("visc", "visc")),

            ]

# version
__id__ = "$Id: Solver.py,v 1.13 2003/08/29 18:06:35 tan2 Exp $"

# End of file
