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


    # for test
    def run(self):
	journal.info("staging").log("setup MPI")
        import mpi
        CitcomModule.citcom_init(mpi.world().handle())

	self.rank = mpi.world().rank
	self.start_time = CitcomModule.CPU_time()
	print "my rank is ", self.rank

        self._setProperties()
        self.prefix = self.inventory.param.inventory.datafile

        if self.inventory.param.inventory.verbose:
            CitcomModule.open_info_file()

        mesher = self.inventory.mesher
        mesher.init(self)

        vsolver = self.inventory.vsolver
        vsolver.init(self)

        tsolver = self.inventory.tsolver
        tsolver.init(self)

        #mesher.run()

        return


    def __init__(self):
        Application.__init__(self, "regional-citcoms")
        #self.total_time = 0
        #self.cycles = 0
        #self.keep_going = True

        return


    def init(self):
        return


    #def fini(self):
	#self.total_time = CitcomModule.CPU_time() - self.start_time
	#CitcomModule.finalize()
	#Application.fini()
	#return


    def output(self):
        import CitcomS.Output as Output
        output_coord = Output.outputCoord(self.prefix, self.rank)
	output_coord.go()

	output_velo = Output.outputVelo(self.prefix, self.rank, self.cycles)
        output_velo.go()

	output_visc = Output.outputVisc(self.prefix, self.rank, self.cycles)
        output_visc.go()

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
        import CitcomS.Advection_diffusion
        import CitcomS.Sphere
	import CitcomS.Stokes_solver

        # components
        from CitcomS.Components.BC import BC
        from CitcomS.Components.Const import Const
        from CitcomS.Components.IC import IC
	from CitcomS.Components.Parallel import Parallel
	from CitcomS.Components.Param import Param
        from CitcomS.Components.Phase import Phase
        from CitcomS.Components.Visc import Visc

        inventory = [
            Mesher("mesher", CitcomS.Sphere.regionalSphere(CitcomModule)),
            VSolver("vsolver", CitcomS.Stokes_solver.imcompressibleNewtonian(CitcomModule)),
            TSolver("tsolver", CitcomS.Advection_diffusion.temperature_diffadv(CitcomModule)),

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


    # this is the true run(), but old
    def run_old(self):
	Application.run(self)

        # read in parameters
        CitcomModule.read_instructions(self.filename)

	#if (Control.post_proccessing):
	#    CitcomModule.post_processing()
	#    return

	# decide which stokes solver to use
	import CitcomS.Stokes_solver
	vsolver = CitcomS.Stokes_solver.imcompressibleNewtionian('imcompressible')
	vsolver.init()

	# decide which field to advect (and to diffuse)
	import CitcomS.Advection_diffusion as Advection_diffusion
	tsolver = Advection_diffusion.temperature_diffadv('temp')
	tsolver.init()


	# solve for 0th time step velocity and pressure
	vsolver.run()

	# output phase
        self.output()

	while (self.keep_going and not self.Emergency_stop):
	    self.cycles += 1
	    print 'cycles = ', self.cycles
	    #tsolver.run()
	    #vsolver.run()
	    total_time = CitcomModule.CPU_time() - self.start_time

            #self.output()
	return


# version
__id__ = "$Id: CitcomSRegional.py,v 1.19 2003/07/24 17:46:46 tan2 Exp $"

# End of file
