#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from mpi.Application import Application
import CitcomS.Regional as CitcomModule
import journal


class RegionalApp(Application):



    def run(self):

        mesher = self.inventory.mesher
        mesher.init(self)

        vsolver = self.inventory.vsolver
        vsolver.init(self)

        tsolver = self.inventory.tsolver
        tsolver.init(self)

	#if (self.invenotry.param.inventory.post_proccessing):
	#    CitcomModule.post_processing()
	#    return

        if self.inventory.param.inventory.verbose:
            CitcomModule.open_info_file()

        mesher.run()

	# solve for 0th time step velocity and pressure
	vsolver.run()

	# output phase
        self._output(self._cycles)

	while (self._keep_going):
	    tsolver.run()
	    vsolver.run()

	    self._cycles += 1
            self._output(self._cycles)

	    if self._cycles >= self.inventory.param.inventory.maxstep:
		self._keep_going = False

        return



    def __init__(self):
        Application.__init__(self, "regional-citcoms")
        return



    def init(self):
	journal.info("staging").log("setup MPI")
        import mpi
        CitcomModule.citcom_init(mpi.world().handle())
	CitcomModule.global_default_values()
        CitcomModule.set_signal()
        self._setProperties()

	self._start_time = CitcomModule.CPU_time()
        self._cycles = 0
        self._keep_going = True

	self.rank = mpi.world().rank
	print "my rank is ", self.rank

        return



    def fini(self):
        total_time = CitcomModule.CPU_time() - self._start_time
        print "Average cpu time taken for velocity step = %f" % (
            total_time / self._cycles )

	#CitcomModule.finalize()
	#Application.fini()

	return



    def _output(self, cycles):
        CitcomModule.output(cycles)
        return



    def _setProperties(self):
	inv = self.inventory

        inv.mesher.setProperties(CitcomModule.mesher_set_properties)
        inv.tsolver.setProperties(CitcomModule.tsolver_set_properties)
        inv.vsolver.setProperties(CitcomModule.vsolver_set_properties)

        inv.bc.setProperties(CitcomModule.BC_set_properties)
        inv.const.setProperties(CitcomModule.Const_set_properties)
        inv.ic.setProperties(CitcomModule.IC_set_properties)
        inv.parallel.setProperties(CitcomModule.Parallel_set_properties)
        inv.param.setProperties(CitcomModule.Param_set_properties)
        inv.phase.setProperties(CitcomModule.Phase_set_properties)
        inv.visc.setProperties(CitcomModule.Visc_set_properties)

        return



    class Inventory(Application.Inventory):

        import pyre.facilities

        # facilities
        from CitcomS.Facilities.Mesher import Mesher
        from CitcomS.Facilities.TSolver import TSolver
	from CitcomS.Facilities.VSolver import VSolver

        # component modules
        import CitcomS.Components.Advection_diffusion as Advection_diffusion
        import CitcomS.Components.Sphere as Sphere
	import CitcomS.Components.Stokes_solver as Stokes_solver

        # components
        from CitcomS.Components.BC import BC
        from CitcomS.Components.Const import Const
        from CitcomS.Components.IC import IC
	from CitcomS.Components.Parallel import Parallel
	from CitcomS.Components.Param import Param
        from CitcomS.Components.Phase import Phase
        from CitcomS.Components.Visc import Visc

        inventory = [
            Mesher("mesher", Sphere.regionalSphere(CitcomModule)),
            VSolver("vsolver", Stokes_solver.imcompressibleNewtonian(CitcomModule)),
            TSolver("tsolver", Advection_diffusion.temperature_diffadv(CitcomModule)),

            pyre.facilities.facility("bc",
				     default=BC("bc", "bc", CitcomModule)),
            pyre.facilities.facility("const",
				     default=Const("const", "const", CitcomModule)),
            pyre.facilities.facility("ic",
				     default=IC("ic", "ic", CitcomModule)),
	    pyre.facilities.facility("parallel",
				     default=Parallel("parallel", "parallel", CitcomModule)),
            pyre.facilities.facility("param",
				     default=Param("param", "param", CitcomModule)),
            pyre.facilities.facility("phase",
				     default=Phase("phase", "phase", CitcomModule)),
            pyre.facilities.facility("visc",
				     default=Visc("visc", "visc", CitcomModule)),

            ]



# version
__id__ = "$Id: CitcomSRegional.py,v 1.24 2003/07/28 23:03:48 tan2 Exp $"

# End of file
