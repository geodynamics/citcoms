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

        self.coupler = None
        self.exchanger = None
        self.myPlus = []
        self.remotePlus = []

	self.start_cpu_time = 0
        self.cpu_time = 0
        self.model_time = 0
        self.fptime = None
        return



    def initialize(self, application):
        BaseSolver.initialize(self, application)

        comm = application.solverCommunicator
        self.all_variables = self.CitcomModule.citcom_init(comm.handle())
	self.communicator = comm

        # information about clock time
	self.start_cpu_time = self.CitcomModule.CPU_time()
        self.cpu_time = self.start_cpu_time
        self.fptime = open("%s.time" % self.inventory.datafile, "w")

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

        self.restart = self.inventory.ic.inventory.restart
        self.ic_initTemperature = self.inventory.ic.initTemperature

        # if there is a coupler, initialize it
        try:
            application.inventory.coupler
        except AttributeError:
            pass
        else:
            self.myPlus = application.myPlus
            self.remotePlus = application.remotePlus
            self.exchanger = application.exchanger
            self.coupler = application.inventory.coupler
            self.coupler.initialize(self)

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

        ic = self.inventory.ic

        # if there is a coupler, launch it
        if self.coupler:
            self.coupler.launch(self)
            # switch the default initTemperature to couled version
            ic.initTemperature = self.exchanger.initTemperature

        # initial conditions
        ic.launch()

        # initialize const. related to mesh
        vsolver.launch()
        tsolver.launch()

        self.solveVelocities()

        return



    def solveVelocities(self):
        vsolver = self.inventory.vsolver
        if self.coupler:
            self.coupler.preVSolverRun()
            vsolver.run()
            self.coupler.postVSolverRun()
        else:
            vsolver.run()
        return



    def solveTemperature(self, dt):
        tsolver = self.inventory.tsolver
        tsolver.run(dt)
        return



    def newStep(self, t, step):
        BaseSolver.newStep(self, t, step)
        if self.coupler:
            self.coupler.newStep()
        return



    #def applyBoundaryConditions(self):
        #BaseSolver.applyBoundaryConditions(self)
        #if self.coupler:
        #    self.coupler.applyBoundaryConditions()
        #return



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

        self.solveTemperature(dt)
        self.solveVelocities()

        return



    def endTimestep(self, t, steps, done):
        BaseSolver.endTimestep(self, t)

        self.inventory.bc.updateBC()
        if self.coupler:
            done = self.coupler.endTimestep(done)

        return done


    def endSimulation(self, step):
        BaseSolver.endSimulation(self, step, self.t)

        total_cpu_time = self.CitcomModule.CPU_time() - self.start_cpu_time

        rank = self.communicator.rank
        if not rank:
            print "Average cpu time taken for velocity step = %f" % (
                total_cpu_time / step )

	#self.CitcomModule.finalize()
        return



    def save(self, step):
        self.CitcomModule.output(self.all_variables, step)
        return


    def timesave(self, t, steps):
        # output time information
        time = self.CitcomModule.CPU_time()
        msg = "%d %.4e %.4e %.4e %.4e" % (steps,
                                          t,
                                          t - self.model_time,
                                          time - self.start_cpu_time,
                                          time - self.cpu_time)
        print >> self.fptime, msg
        self.fptime.flush()

        self.model_time = t
        self.cpu_time = time
        return


    def setProperties(self):
        self.CitcomModule.Solver_set_properties(self.all_variables,
                                                self.inventory)

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

            TSolver("tsolver", default=Advection_diffusion.temperature_diffadv()),
            VSolver("vsolver", default=Stokes_solver.incompressibleNewtonian()),

            pyre.facilities.facility("bc", default=BC()),
            pyre.facilities.facility("const", default=Const()),
            pyre.facilities.facility("ic", default=IC()),
            pyre.facilities.facility("param", default=Param()),
            pyre.facilities.facility("phase", default=Phase()),
            pyre.facilities.facility("visc", default=Visc()),

            pyre.properties.float("rayleigh", default=1e+08),
            pyre.properties.float("Q0", default=0.0),

            pyre.properties.bool("stokes_flow_only", default=False),

            pyre.properties.bool("verbose", default=False),
            pyre.properties.bool("see_convergence", default=True),

            ]

# version
__id__ = "$Id: Solver.py,v 1.38 2004/05/24 20:32:06 tan2 Exp $"

# End of file
