#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.components.Solver import Solver
import journal


class Citcom(Solver):


    def run(self):
        self.startSimulation()
        self.run_simulation()
        self.endSimulation()
        return



    def startSimulation(self):
	#journal.info("staging").log("setup MPI")
        comm = self.get_communicator()

        self.all_variables = self.CitcomModule.citcom_init(comm.handle())

        self.initialize()
	self.CitcomModule.global_default_values(self.all_variables)
        self.CitcomModule.set_signal()
        self.setProperties()

	self._start_cpu_time = self.CitcomModule.CPU_time()
        self._time = 0
        self._cycles = 0

	self.rank = comm.rank
	print "my rank is ", self.rank
        return



    def run_simulation(self):

        mesher = self.inventory.mesher
        mesher.setup()

        vsolver = self.inventory.vsolver
        vsolver.setup()

        tsolver = self.inventory.tsolver
        tsolver.setup()

	#if (self.invenotry.param.inventory.post_proccessing):
	#    self.CitcomModule.post_processing()
	#    return

        mesher.run()

	# solve for 0th time step velocity and pressure
	vsolver.run()

        self.save(self._cycles)

	while self._cycles < self.inventory.param.inventory.maxstep:

	    #tsolver.run()
            if not self._cycles:
                self.CitcomModule.PG_timestep_init(self.all_variables)

            dt = tsolver.stable_timestep()
            self.CitcomModule.PG_timestep_solve(self.all_variables, dt)

	    vsolver.run()

            self._time += dt
	    self._cycles += 1

            if not (self._cycles %
                    self.inventory.param.inventory.storage_spacing):
                self.save(self._cycles)


        return



    def endSimulation(self):
        total_cpu_time = self.CitcomModule.CPU_time() - self._start_cpu_time
        if not self.rank:
            print "Average cpu time taken for velocity step = %f" % (
                total_cpu_time / self._cycles )

	#self.CitcomModule.finalize()

        return



    def get_communicator(self):
	#journal.info("staging").log("setup MPI")
        import mpi
        world = mpi.world()

        if self.inventory.ranklist:
            comm = world.include(self.inventory.ranklist)
            return comm
        else:
            return world



    def save(self, cycles):
        self.CitcomModule.output(self.all_variables, cycles)
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



    class Inventory(Solver.Inventory):

        import pyre.facilities
        import pyre.properties

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

            pyre.properties.sequence("ranklist", []),

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
__id__ = "$Id: Solver.py,v 1.6 2003/08/27 20:52:46 tan2 Exp $"

# End of file
