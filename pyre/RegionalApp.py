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


class CitcomsRegionalApp(Application):


    def run(self):
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


    def __init__(self):
        Application.__init__(self, "citcomsregional")
        return


    class Facilities(Application.Facilities):


        import pyre.facilities
        from EarthModelConstants import EarthModelConstants
        from EarthModelGrid import EarthModelGrid
        from EarthModelPhase import EarthModelPhase
        from EarthModelVisc import EarthModelVisc
        
        __facilities__ = Application.Facilities.__facilities__ + (
            pyre.facilities.facility("earthModel", EarthModelConstants()),
            pyre.facilities.facility("earthModel_grid", EarthModelGrid()),
            pyre.facilities.facility("earthModel_phase", EarthModelPhase()),
            pyre.facilities.facility("earthModel_visc", EarthModelVisc()),
            )


    class Properties(Application.Properties):


        __properties__ = Application.Properties.__properties__ + (
            )


# version
__id__ = "$Id: RegionalApp.py,v 1.1 2003/04/09 18:59:23 ces74 Exp $"

# End of file 
