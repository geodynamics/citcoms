#! /usr/bin/env python
try:
    import wxversion
    wxversion.ensureMinimal('2.6')
except ImportError:
    pass

from enthought.mayavi.app import Mayavi
import sys
from os.path import isfile
from getopt import getopt, GetoptError

class HdfDisplay(Mayavi):
    
    filename = None
    timestep = 0
    nx_redu = 0
    ny_redu = 0
    nz_redu = 0
    
    def run(self):

        from enthought.mayavi.sources.vtk_file_reader import VTKFileReader
        #import modules here
        from enthought.mayavi.modules import surface, glyph , axes, outline, orientation_axes, scalar_cut_plane  
        from enthought.mayavi.sources.vtk_data_source import VTKDataSource 
        from enthought.tvtk.api import tvtk
        #citcomS Filter and Modules
        from plugins.CitcomSHDFUgrid import CitcomSHDFUgrid
        from plugins.filter.CitcomSshowCaps import CitcomSshowCaps
        from plugins.filter.CitcomSreduce import CitcomSreduce
        
        import re
        
        
        script = self.script
         
        #DEFINES
        orange = (1.0,0.5,0)
        reduce_factor = 2
        
        
        #Read Hdf file
        src_hdf = CitcomSHDFUgrid()
        hexgrid = src_hdf.initialize(self.filename,self.timestep,self.nx_redu,self.ny_redu,self.nz_redu)
        radius_inner = src_hdf._radius_inner
        data = VTKDataSource()
        data.data = hexgrid
        
        
        ###########Display Data############
        #Create new scene
        script.new_scene()     
        script.add_source(data)
             
        scap = CitcomSshowCaps()
        script.add_filter(scap)
        
        
        #Show ScalarCutPlane
        scp = scalar_cut_plane.ScalarCutPlane()
        script.add_module(scp)
        
        #Add filter for a reduce grid
        redu = CitcomSreduce()
        #redu.setvalues(nx,ny,nz)
        script.add_filter(redu)
       
        gly = glyph.Glyph()
        gly.glyph.glyph_source.scale = 0.082
        gly.glyph.scale_mode = 'scale_by_scalar'
        gly.glyph.color_mode = 'color_by_scalar'
        script.add_module(gly)
        mm = gly.module_manager
        mm.scalar_lut_manager.use_default_range = False
        mm.scalar_lut_manager.data_range = 0.0, 1.0
        ################### Create CORE ################################
        #Load VTK Data Sets
        sphere = tvtk.SphereSource()
        sphere.radius = radius_inner 
        sphere.theta_resolution = 24 
        sphere.phi_resolution = 24
          
        # Create a mesh from the data created above.
        src = VTKDataSource()
        src.data = sphere.output
        script.add_source(src)
        
        #Show Surface
        surf_module = surface.Surface()
        surf_module.actor.property.color = orange
        script.add_module(surf_module)
        
        
         # to create the rendering scene
         ## your stuff here

if __name__ == '__main__':
    mc = HdfDisplay()
    if len(sys.argv)>=3:
        mc.filename = sys.argv[1]
        try:
            mc.timestep = int(sys.argv[2])
        except ValueError:
            print "Timestep is not a number."
            sys.exit(1)
        if not isfile(mc.filename):
            print "File not found."
            sys.exit(1)
            
    else:
        print "[filename] [timestep] -x [Reduce Grid Size X] -y [Reduce Grid Size X] -z [Reduce Grid Size Z]"
        sys.exit(0)
   ##parse for reduction factors 
    try:
        opts, args = getopt(sys.argv[3:], "x:y:z:", ['x=','y=','z='])
    except GetoptError, msg:
        print "Error: %s" % msg
        sys.exit(1)
    
    for opt,arg in opts:
        if opt in ('-x','--nx_redu'):
            try:
                mc.nx_redu = int(arg)
                print "Reducing Grid Size to x:",mc.nx_redu
            except ValueError:
                print "x is not a number..."
                sys.exit(1)
                
        if opt in ('-y','--ny_redu'):
            try:
                mc.ny_redu = int(arg)
                print "Reducing Grid Size to y:",mc.ny_redu
            except ValueError:
                print "y is not a number..."
                sys.exit(1)
        
        if opt in ('-z','--nz_redu'):
            try:
                mc.nz_redu = int(arg)
                print "Reducing Grid Size to z:",mc.nz_redu
        
            except ValueError:
                print "z is not a number..."
                sys.exit(1)
        
    mc.main()
