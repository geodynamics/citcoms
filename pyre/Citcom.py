#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.components.Component import Component
import journal


class Citcom(Component):


    def __init__(self, name):
        Component.__init__(self, "citcom", name)
        return



    def run(self):
        self.start_simulation()
        self.run_simulation()
        self.end_simulation()
        return



    def start_simulation(self):
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

        self._output(self._cycles)

	while self._cycles < self.inventory.param.inventory.maxstep:
	    self._cycles += 1

	    tsolver.run()
	    vsolver.run()

            if not (self._cycles %
                    self.inventory.param.inventory.storage_spacing):
                self._output(self._cycles)


        return



    def end_simulation(self):
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



    class Inventory(Component.Inventory):

        import pyre.properties

        inventory = [

            pyre.properties.sequence("ranklist", []),

            ]

# version
__id__ = "$Id: Citcom.py,v 1.5 2003/08/25 19:16:04 tan2 Exp $"

# End of file
