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


class RegionalApp(Application):

    #import journal


    # for test
    def run(self):
	Application.run(self)

	self.facilities.bc.setProperties()
	return

	#self.test_facility()
	vsolver = self.facilities.vsolver
	vsolver.run()
	return



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
	print self.filename

        return


    def preInit(self):
        import mpi

        Application.preInit(self)
        Regional.Citcom_Init(mpi.world().handle())

	self.rank = mpi.world().rank
	self.start_time = Regional.CPU_time()
        #print self.start_time

	return



    def fini(self):
	self.total_time = Regional.CPU_time() - self.start_time
	Regional.finalize()
	Application.fini()
	return


    def output(self):
        import CitcomS.Output as Output
        output_coord = Output.outputCoord(self.prefix, self.rank)
	output_coord.go()

	output_velo = Output.outputVelo(self.prefix, self.rank, self.cycles)
        output_velo.go()

	output_visc = Output.outputVisc(self.prefix, self.rank, self.cycles)
        output_visc.go()

        return



    class Facilities(Application.Facilities):

        import pyre.facilities

	from CitcomS.Facilities.VSolver import VSolver
	import CitcomS.Stokes_solver

        from CitcomS.Components.BC import BC
        from CitcomS.Components.Const import Const
        from CitcomS.Components.IC import IC
        from CitcomS.Components.Mesh import Mesh
	from CitcomS.Components.Parallel import Parallel
	from CitcomS.Components.Param import Param
        from CitcomS.Components.Phase import Phase
        from CitcomS.Components.Visc import Visc

        __facilities__ = Application.Facilities.__facilities__ + (
            VSolver("vsolver", CitcomS.Stokes_solver.imcompressibleNewtonian()),

            pyre.facilities.facility("bc", BC()),
            pyre.facilities.facility("const", Const()),
            pyre.facilities.facility("ic", IC()),
            pyre.facilities.facility("mesh", Mesh()),
	    pyre.facilities.facility("parallel", Parallel()),
            pyre.facilities.facility("param", Param()),
            pyre.facilities.facility("phase", Phase()),
            pyre.facilities.facility("visc", Visc()),

            )


    class Properties(Application.Properties):


        __properties__ = Application.Properties.__properties__ + (
            )


    def test_facility(self):
        import mpi
        import CitcomS.RadiusDepth

        world = mpi.world()
        rank = world.rank
        size = world.size

        print "Hello from [%d/%d]" % (world.rank, world.size)

        const = self.facilities.const.properties
        mesh = self.facilities.mesh.properties
        phase = self.facilities.phase.properties
        visc = self.facilities.visc.properties

        print "%02d: Constants:" % rank
        print "%02d:      const.radius: %s" % (rank, const.radius)
        print "%02d:      const.ref_density: %s" % (rank, const.ref_density)
        print "%02d:      const.thermdiff: %s" % (rank, const.thermdiff)
        print "%02d:      const.gravacc: %s" % (rank, const.gravacc)
        print "%02d:      const.thermexp: %s" % (rank, const.thermexp)
        print "%02d:      const.ref_visc: %s" % (rank, const.ref_visc)
        print "%02d:      const.heatcapacity: %s" % (rank, const.heatcapacity)
        print "%02d:      const.water_density: %s" % (rank, const.water_density)
        print "%02d:      const.depth_lith: %s" % (rank, const.depth_lith)
        print "%02d:      const.depth_410: %s" % (rank, const.depth_410)
        print "%02d:      const.depth_660: %s" % (rank, const.depth_660)
        print "%02d:      const.depth_cmb: %s" % (rank, const.depth_cmb)
        print "%02d: Mesh:" % rank
        print "%02d:      mesh.coord: %s" % (rank, mesh.coord)
        print "%02d:      mesh.coord_file: %s" % (rank, mesh.coord_file)
        print "%02d:      mesh.nodex: %s" % (rank, mesh.nodex)
        print "%02d:      mesh.mgunitx: %s" % (rank, mesh.mgunitx)
        print "%02d:      mesh.levels: %s" % (rank, mesh.levels)
        print "%02d:      mesh.theta_min: %s" % (rank, mesh.theta_min)
        print "%02d:      mesh.phi_min: %s" % (rank, mesh.phi_min)
        print "%02d:      mesh.radius_innter: %s" % (rank, mesh.radius_inner)
        print "%02d: PhaseChange:" % rank
        print "%02d:      phase.Ra410: %s" % (rank, phase.Ra_410)
        print "%02d:      phase.clapeyron410: %s" % (rank, phase.clapeyron410)
        print "%02d:      phase.transT410: %s" % (rank, phase.transT410)
        print "%02d:      phase.width410: %s" % (rank, phase.width410)
        print "%02d: Viscosity:" % rank
        print "%02d:      visc.Viscosity: %s" % (rank, visc.Viscosity)
        print "%02d:      visc.rheol: %s" % (rank, visc.rheol)
        print "%02d:      visc.visc_smooth_method: %s" % (rank, visc.visc_smooth_method)
        print "%02d:      visc.VISC_UPDATE: %s" % (rank, visc.VISC_UPDATE)
        print "%02d:      visc.viscE: %s" % (rank, visc.viscE)
        print "%02d:      visc.viscT: %s" % (rank, visc.viscT)
        print "%02d:      visc.visc0: %s" % (rank, visc.visc0)
        return


# version
__id__ = "$Id: RegionalSolver.py,v 1.9 2003/06/27 00:15:02 tan2 Exp $"

# End of file
