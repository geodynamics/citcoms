#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from mpi.Application import Application
import CitcomS.Regional as Regional
import journal


class RegionalApp(Application):


    # for test
    def run(self):
	journal.info("staging").log("setup MPI")
        import mpi
        Regional.Citcom_Init(mpi.world().handle())

	self.rank = mpi.world().rank
	self.start_time = Regional.CPU_time()
	print "my rank is ", self.rank

        vsolver = self.inventory.vsolver
        print vsolver
        #tsolver = self.inventory.tsolver

        return




    def __init__(self, inputfile):
        Application.__init__(self, "citcomsregional")
	self.filename = inputfile
        self.total_time = 0
        self.cycles = 0
        self.keep_going = True
        self.Emergency_stop = False
        self.start_time = 0.0

	#test
	self.prefix = 'test'

        return



    #def fini(self):
	#self.total_time = Regional.CPU_time() - self.start_time
	#Regional.finalize()
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



    class Inventory(Application.Inventory):

        import pyre.facilities

	from CitcomS.Facilities.VSolver import VSolver
        from CitcomS.Facilities.TSolver import TSolver

	import CitcomS.Stokes_solver
        import CitcomS.Advection_diffusion

        from CitcomS.Components.BC import BC
        from CitcomS.Components.Const import Const
        from CitcomS.Components.IC import IC
        from CitcomS.Components.Mesh import Mesh
	from CitcomS.Components.Parallel import Parallel
	from CitcomS.Components.Param import Param
        from CitcomS.Components.Phase import Phase
        from CitcomS.Components.Visc import Visc

        inventory = [
            VSolver("vsolver", CitcomS.Stokes_solver.imcompressibleNewtonian()),

            #TSolver("tsolver", default=CitcomS.Advection_diffusion.temperature_diffadv()),

            pyre.facilities.facility("bc", default=BC()),
            pyre.facilities.facility("const", default=Const()),
            pyre.facilities.facility("ic", default=IC()),
            pyre.facilities.facility("mesh", default=Mesh()),
	    pyre.facilities.facility("parallel", default=Parallel()),
            pyre.facilities.facility("param", default=Param()),
            pyre.facilities.facility("phase", default=Phase()),
            pyre.facilities.facility("visc", default=Visc()),

            ]


    # this is the true run(), but old
    def run_old(self):
	Application.run(self)

        # read in parameters
        Regional.read_instructions(self.filename)

	#if (Control.post_proccessing):
	#    Regional.post_processing()
	#    return

	# decide which stokes solver to use
	import CitcomS.Stokes_solver
	vsolver = CitcomS.Stokes_solver.imcompressibleNewtionian('imcompressible')
	vsolver.init()

	# decide which field to advect (and to diffuse)
	import CitcomS.Advection_diffusion.Advection_diffusion as Advection_diffusion
	tsolver = Advection_diffusion.PG_timestep('temp')
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
	    total_time = Regional.CPU_time() - self.start_time

            #self.output()
	return


# version
__id__ = "$Id: CitcomSRegional.py,v 1.12 2003/07/09 19:42:27 tan2 Exp $"

# End of file
