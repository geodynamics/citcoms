#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2003  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from mpi.Application import Application
import CitcomS.Regional as Regional


class RegionalApp(Application):
#    import journal
    total_time = 0
    cycles = 0
    keep_going = True
    Emergency_stop = False

    def run(self):
	Application.run(self)
	
#	if (Control.post_proccessing):
#	    Regional.post_processing()
#	    return
	
	# decide which stokes solver to use
	import CitcomS.Stokes_solver
	vsolver = CitcomS.Stokes_solver.imcompressibleNewtionian('imcompressible')
	vsolver.init()
	
#	# decide which advection solver to use
#	import CitcomS.Advection_solver
#	tsolver = CitcomS.Advection_solver.PGsolver()
#	tsolver.init()

	#
	#import CitcomS.Regional.output as Output
	#Output.init()

	# solve for 0th time step velocity and pressure
	vsolver.run()

#	while (self.keep_going and not Emergency_stop):
#	    self.cycles += 1
#	    tsolver.run()
#	    vsolver.run()
#	    total_time = Regional.CPU_time() - self.start_time

	return
	    

    def __init__(self, inputfile):
        Application.__init__(self, "citcomsregional")
	self.filename = inputfile
	print self.filename
        return


    def preInit(self):
        import mpi
        #world = mpi.world()
        Application.preInit(self)
        Regional.Citcom_Init(mpi.mpi.world)
	self.start_time = Regional.CPU_time()
	return
 

    def postInit(self):
        import sys
        Application.postInit(self)
        #filename = self.facility.infile
        Regional.read_instructions(self.filename)
	return
    

    def fini(self):
	total_time = Regional.CPU_time() - self.start_time
	Regional.finalize()
	Application.fini()
	return


    class Facilities(Application.Facilities):

        import pyre.facilities
        from EarthModelConstants import EarthModelConstants
        from EarthModelPhase import EarthModelPhase
        from EarthModelVisc import EarthModelVisc
        from SimulationGrid import SimulationGrid
        
        __facilities__ = Application.Facilities.__facilities__ + (
            pyre.facilities.facility("earthModel", EarthModelConstants()),
            pyre.facilities.facility("earthModel_phase", EarthModelPhase()),
            pyre.facilities.facility("earthModel_visc", EarthModelVisc()),
            pyre.facilities.facility("simulation_grid", SimulationGrid()),
            )


    class Properties(Application.Properties):


        __properties__ = Application.Properties.__properties__ + (
            )


    def test(self):
        import mpi
        import CitcomS.RadiusDepth
        
        world = mpi.world()
        rank = world.rank
        size = world.size
        
        print "Hello from [%d/%d]" % (world.rank, world.size)

        earthProps = self.facilities.earthModel.properties
        earthGrid = self.facilities.earthModel_grid.properties
        earthPhase = self.facilities.earthModel_phase.properties        
        earthVisc = self.facilities.earthModel_visc.properties
        
        print "%02d: EarthModelConstants:" % rank
        print "%02d:      EarthModel.radius: %s" % (rank, earthProps.radius)
        print "%02d:      EarthModel.ref_density: %s" % (rank, earthProps.ref_density)
        print "%02d:      EarthModel.thermdiff: %s" % (rank, earthProps.thermdiff)
        print "%02d:      EarthModel.gravacc: %s" % (rank, earthProps.gravacc)
        print "%02d:      EarthModel.thermexp: %s" % (rank, earthProps.thermexp)
        print "%02d:      EarthModel.ref_visc: %s" % (rank, earthProps.ref_visc)
        print "%02d:      EarthModel.heatcapacity: %s" % (rank, earthProps.heatcapacity)
        print "%02d:      EarthModel.water_density: %s" % (rank, earthProps.water_density)
        print "%02d:      EarthModel.depth_lith: %s" % (rank, earthProps.depth_lith)
        print "%02d:      EarthModel.depth_410: %s" % (rank, earthProps.depth_410)
        print "%02d:      EarthModel.depth_660: %s" % (rank, earthProps.depth_660)
        print "%02d:      EarthModel.depth_cmb: %s" % (rank, earthProps.depth_cmb)
        print "%02d: EarthModelGrid:" % rank
        print "%02d:      EarthModel.grid.coor: %s" % (rank, earthGrid.coor)
        print "%02d:      EarthModel.grid.coor_file: %s" % (rank, earthGrid.coor_file)
        print "%02d:      EarthModel.grid.nodex: %s" % (rank, earthGrid.nodex)
        print "%02d:      EarthModel.grid.mgunitx: %s" % (rank, earthGrid.mgunitx)
        print "%02d:      EarthModel.grid.levels: %s" % (rank, earthGrid.levels)
        print "%02d:      EarthModel.grid.theta_min: %s" % (rank, earthGrid.theta_min)
        print "%02d:      EarthModel.grid.fi_min: %s" % (rank, earthGrid.fi_min)
        print "%02d:      EarthModel.grid.radius_innter: %s" % (rank, earthGrid.radius_inner)
        print "%02d: EarthModelPhase:" % rank
        print "%02d:      EarthModel.phase.Ra410: %s" % (rank, earthPhase.Ra_410)
        print "%02d:      EarthModel.phase.clapeyron410: %s" % (rank, earthPhase.clapeyron410)
        print "%02d:      EarthModel.phase.transT410: %s" % (rank, earthPhase.transT410)
        print "%02d:      EarthModel.phase.width410: %s" % (rank, earthPhase.width410)
        print "%02d: EarthModelVisc:" % rank
        print "%02d:      EarthModel.visc.Viscosity: %s" % (rank, earthVisc.Viscosity)
        print "%02d:      EarthModel.visc.rheol: %s" % (rank, earthVisc.rheol)
        print "%02d:      EarthModel.visc.visc_smooth_method: %s" % (rank, earthVisc.visc_smooth_method)
        print "%02d:      EarthModel.visc.VISC_UPDATE: %s" % (rank, earthVisc.VISC_UPDATE)
        print "%02d:      EarthModel.visc.viscE: %s" % (rank, earthVisc.viscE)
        print "%02d:      EarthModel.visc.viscT: %s" % (rank, earthVisc.viscT)
        print "%02d:      EarthModel.visc.visc0: %s" % (rank, earthVisc.visc0)        
        return


# version
__id__ = "$Id: CitcomSRegional.py,v 1.4 2003/05/16 21:11:54 tan2 Exp $"

# End of file 
