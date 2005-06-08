#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.simulations.Solver import Solver as BaseSolver
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
        inv.tracer.initialize(CitcomModule, all_variables)
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

            if not (ic.inventory.restart or ic.inventory.post_p):
                # switch the default initTemperature to coupled version
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



    def solveAdditional(self):
        if not self.coupler:
            # tracer module doesn't work with exchanger module
            self.inventory.tracer.run()
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
        self.solveAdditional()

        return



    def endTimestep(self, t, steps, done):
        BaseSolver.endTimestep(self, t)

        self.inventory.visc.updateMaterial()
        self.inventory.bc.updatePlateVelocity()

        if self.coupler:
            done = self.coupler.endTimestep(steps, done)

        return done


    def endSimulation(self, step):
        BaseSolver.endSimulation(self, step, self.t)

        total_cpu_time = self.CitcomModule.CPU_time() - self.start_cpu_time

        rank = self.communicator.rank
        if not rank:
            print "Average cpu time taken for velocity step = %f" % (
                total_cpu_time / step )

        if self.coupler:
            self.CitcomModule.output(self.all_variables, step)

	#self.CitcomModule.finalize()
        return



    def save(self, step, monitoringFrequency):
        # for non-coupled run, output spacing is 'monitoringFrequency'
        if not (step % monitoringFrequency):
            self.CitcomModule.output(self.all_variables, step)
        elif self.coupler and not (self.coupler.exchanger.coupled_steps % monitoringFrequency):
            print self.coupler.exchanger.coupled_steps, monitoringFrequency
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
        inv.tracer.setProperties()
        inv.visc.setProperties()

        return



    class Inventory(BaseSolver.Inventory):

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

        rayleigh = pyre.inventory.float("rayleigh", default=1e+05)
        Q0 = pyre.inventory.float("Q0", default=0.0)

        stokes_flow_only = pyre.inventory.bool("stokes_flow_only", default=False)

        verbose = pyre.inventory.bool("verbose", default=False)
        see_convergence = pyre.inventory.bool("see_convergence", default=True)


# version
__id__ = "$Id: Solver.py,v 1.45 2005/06/08 01:55:34 leif Exp $"

# End of file
