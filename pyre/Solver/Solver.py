#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from mpi.Application import Application
import journal


class CitcomApp(Application):


    def run(self):
	#journal.info("staging").log("setup MPI")
        comm = self.get_communicator()

        E = self.CitcomModule.citcom_init(comm.handle())
        self.all_variables = E
	self.CitcomModule.global_default_values(self.all_variables)
        self.CitcomModule.set_signal()
        self._setProperties()

	self._start_time = self.CitcomModule.CPU_time()
        self._cycles = 0

	self.rank = comm.rank
	print "my rank is ", self.rank

        mesher = self.inventory.mesher
        mesher.init(self)

        vsolver = self.inventory.vsolver
        vsolver.init(self)

        tsolver = self.inventory.tsolver
        tsolver.init(self)

	#if (self.invenotry.param.inventory.post_proccessing):
	#    self.CitcomModule.post_processing()
	#    return

        mesher.run()

	# solve for 0th time step velocity and pressure
	vsolver.run()

        self._output(self._cycles)

	while self._cycles < self.inventory.param.inventory.maxstep:
	    self._cycles += 1

	    tsolver.run()
	    vsolver.run()

            if not (self._cycles %
                    self.inventory.param.inventory.storage_spacing):
                self._output(self._cycles)


        total_time = self.CitcomModule.CPU_time() - self._start_time
        if not self.rank:
            print "Average cpu time taken for velocity step = %f" % (
                total_time / self._cycles )

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



    def init(self):
        return



    def fini(self):
        return



    def _output(self, cycles):
        self.CitcomModule.output(self.all_variables, cycles)
        return



    def _setProperties(self):
	inv = self.inventory

        inv.mesher.setProperties(self.all_variables,
                                 self.CitcomModule.mesher_set_properties)
        inv.tsolver.setProperties(self.all_variables,
                                  self.CitcomModule.tsolver_set_properties)
        inv.vsolver.setProperties(self.all_variables,
                                  self.CitcomModule.vsolver_set_properties)

        inv.bc.setProperties(self.all_variables,
                             self.CitcomModule.BC_set_properties)
        inv.const.setProperties(self.all_variables,
                                self.CitcomModule.Const_set_properties)
        inv.ic.setProperties(self.all_variables,
                             self.CitcomModule.IC_set_properties)
        inv.param.setProperties(self.all_variables,
                                self.CitcomModule.Param_set_properties)
        inv.phase.setProperties(self.all_variables,
                                self.CitcomModule.Phase_set_properties)
        inv.visc.setProperties(self.all_variables,
                               self.CitcomModule.Visc_set_properties)

        return



    class Inventory(Application.Inventory):

        import pyre.facilities
        import pyre.properties

        # facilities
        from Facilities.TSolver import TSolver
	from Facilities.VSolver import VSolver

        # component modules
        import Components.Advection_diffusion as Advection_diffusion
	import Components.Stokes_solver as Stokes_solver

        # components
        from Components.BC import BC
        from Components.Const import Const
        from Components.IC import IC
	from Components.Param import Param
        from Components.Phase import Phase
        from Components.Visc import Visc

        inventory = [

            VSolver("vsolver", Stokes_solver.incompressibleNewtonian(CitcomModule)),
            TSolver("tsolver", Advection_diffusion.temperature_diffadv(CitcomModule)),

            pyre.facilities.facility("bc",
				     default=BC("bc", "bc", CitcomModule)),
            pyre.facilities.facility("const",
				     default=Const("const", "const", CitcomModule)),
            pyre.facilities.facility("ic",
				     default=IC("ic", "ic", CitcomModule)),
            pyre.facilities.facility("param",
				     default=Param("param", "param", CitcomModule)),
            pyre.facilities.facility("phase",
				     default=Phase("phase", "phase", CitcomModule)),
            pyre.facilities.facility("visc",
				     default=Visc("visc", "visc", CitcomModule)),

            pyre.properties.sequence("ranklist", []),

            ]

# version
__id__ = "$Id: Solver.py,v 1.3 2003/08/22 22:18:41 tan2 Exp $"

# End of file
