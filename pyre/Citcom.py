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




    def init(self):
	#journal.info("staging").log("setup MPI")
        import mpi
        self.CitcomModule.citcom_init(mpi.world().handle())
	self.CitcomModule.global_default_values()
        self.CitcomModule.set_signal()
        self._setProperties()

	self._start_time = self.CitcomModule.CPU_time()
        self._cycles = 0
        self._keep_going = True

	self.rank = mpi.world().rank
	print "my rank is ", self.rank

        return



    def fini(self):
        total_time = self.CitcomModule.CPU_time() - self._start_time
        if not self.rank:
            print "Average cpu time taken for velocity step = %f" % (
                total_time / self._cycles )

	#self.CitcomModule.finalize()
	#Application.fini(self)

	return



    def _output(self, cycles):
        self.CitcomModule.output(cycles)
        return



    def _setProperties(self):
	inv = self.inventory

        inv.mesher.setProperties(self.CitcomModule.mesher_set_properties)
        inv.tsolver.setProperties(self.CitcomModule.tsolver_set_properties)
        inv.vsolver.setProperties(self.CitcomModule.vsolver_set_properties)

        inv.bc.setProperties(self.CitcomModule.BC_set_properties)
        inv.const.setProperties(self.CitcomModule.Const_set_properties)
        inv.ic.setProperties(self.CitcomModule.IC_set_properties)
        inv.param.setProperties(self.CitcomModule.Param_set_properties)
        inv.phase.setProperties(self.CitcomModule.Phase_set_properties)
        inv.visc.setProperties(self.CitcomModule.Visc_set_properties)

        return



    class Inventory(Application.Inventory):

	inventory = [ ]

# version
__id__ = "$Id: Citcom.py,v 1.1 2003/08/01 22:24:00 tan2 Exp $"

# End of file
