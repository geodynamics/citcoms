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


    def __init__(self, name, facility="solver"):
        BaseSolver.__init__(self, name, facility)

        self.CitcomModule = None
        self.all_variables = None
        self.communicator = None
        self.intercomm = None
        self.coupler = None

        self.leaderRank = 0
        self.localLeader = 0
        self.remoteLeader = 0
	self._start_cpu_time = 0
        return



    def initialize(self, application):
        BaseSolver.initialize(self, application)
	#journal.info("staging").log("setup MPI")
        comm = application.solverCommunicator
        self.all_variables = self.CitcomModule.citcom_init(comm.handle())
	self.communicator = comm

        # if there is a coupler, initialize it
        if application.inventory.coupler:
            self.intercomm = application.intercomm
            self.leaderRank = application.leaderRank
            self.localLeader = application.localLeader
            self.remoteLeader = application.remoteLeader
            self.coupler = application.inventory.coupler
            self.coupler.initialize(self)

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

        CitcomModule.global_default_values(self.all_variables)
        CitcomModule.set_signal()

        self.setProperties()

	self._start_cpu_time = self.CitcomModule.CPU_time()
        return



    def launch(self, application):
        BaseSolver.launch(self, application)

        mesher = self.inventory.mesher
        mesher.setup()

        vsolver = self.inventory.vsolver
        vsolver.setup()

        tsolver = self.inventory.tsolver
        tsolver.setup()

        # create mesh
        mesher.run()

        # if there is a coupler, launch it
        if self.coupler:
            self.coupler.launch(self)

        # solver for 0th step velocity
        vsolver.run()

        # initialze const. related to mesh
        tsolver.launch()

        return



    def newStep(self, t, step):
        BaseSolver.newStep(self, t, step)
        if self.coupler:
            self.coupler.newStep()
        return



    def applyBoundaryConditions(self):
        BaseSolver.applyBoundaryConditions(self)
        if self.coupler:
            self.coupler.applyBoundaryConditions()
        return



    def stableTimestep(self):
        tsolver = self.inventory.tsolver
        dt = tsolver.stable_timestep()

        if self.coupler:
            # negotiate with other solver(s)
            dt = self.coupler.stableTimestep(dt)

        BaseSolver.stableTimestep(self, dt)
        return dt



    def advance(self, dt):
        BaseSolver.advance(self, dt)

        vsolver = self.inventory.vsolver
        vsolver.setup()

        tsolver = self.inventory.tsolver
        tsolver.setup()

        tsolver.run(dt)
        vsolver.run()

        return



    def endTimestep(self, t):
        BaseSolver.endTimestep(self, t)

        if self.coupler:
            self.coupler.endTimestep()

        return


    def endSimulation(self, step):
        BaseSolver.endSimulation(self, step, self.t)

        total_cpu_time = self.CitcomModule.CPU_time() - self._start_cpu_time

        rank = self.communicator.rank
        if not rank:
            print "Average cpu time taken for velocity step = %f" % (
                total_cpu_time / step )

	#self.CitcomModule.finalize()
        return



    def save(self, step):
        self.CitcomModule.output(self.all_variables, step)
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

            TSolver("tsolver", default=Advection_diffusion.temperature_diffadv("temp")),
            VSolver("vsolver", default=Stokes_solver.incompressibleNewtonian("incomp-newtonian")),

            pyre.facilities.facility("bc", default=BC("bc", "bc")),
            pyre.facilities.facility("const", default=Const("const", "const")),
            pyre.facilities.facility("ic", default=IC("ic", "ic")),
            pyre.facilities.facility("param", default=Param("param", "param")),
            pyre.facilities.facility("phase", default=Phase("phase", "phase")),
            pyre.facilities.facility("visc", default=Visc("visc", "visc")),

            ]

# version
__id__ = "$Id: Solver.py,v 1.20 2003/09/27 20:24:46 tan2 Exp $"

# End of file
