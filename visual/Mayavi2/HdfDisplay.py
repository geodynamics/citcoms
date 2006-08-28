#! /usr/bin/env python

from enthought.mayavi.app import Mayavi
import sys
from os.path import isfile

class HdfDisplay(Mayavi):
    
    filename = None
    timestep = 0
    
    def run(self):

        from enthought.mayavi.sources.vtk_file_reader import VTKFileReader
        #import modules here
        from enthought.mayavi.modules import surface, glyph , axes, outline, orientation_axes, scalar_cut_plane  
        from enthought.mayavi.sources.vtk_data_source import VTKDataSource 
        from enthought.tvtk.api import tvtk
        from core.CitcomSHDFUgrid import CitcomSHDFUgrid
        from filter.CitcomSshowCaps import CitcomSshowCaps
        from filter.CitcomSreduce import CitcomSreduce
        #import filter.CitcomSSphere
        import re
        
        #DEFINES
        orange = (1.0,0.5,0)
        reduce_factor = 2
        nx = 17
        ny = 17
        nz = 17
        
        #Read Hdf file
        src_hdf = CitcomSHDFUgrid()
        hexgrid = src_hdf.initialize(self.filename,self.timestep,0,0,0,False,False)
        radius_inner = src_hdf._radius_inner
        data = VTKDataSource()
        data.data = hexgrid
        
        
        ###########Display Data############
        #Create new scene
        script.new_scene()     
        script.add_source(data)
             
        scap = CitcomSshowCaps
        #scap.setvalues(nx,ny,nz)
        script.add_filter(scap)
        
        #Orientation Axes
        oa = orientation_axes.OrientationAxes()
        script.add_module(oa)
        
        #Show ScalarCutPlane
        scp = scalar_cut_plane.ScalarCutPlane()
        script.add_module(scp)
        
        #Add filter for a reduce grid
        redu = CitcomSreduce()
        #redu.setvalues(nx,ny,nz)
        script.add_filter(redu)
       
        gly = glyph.Glyph()
        gly.glyph.glyph_source.scale = 0.082
        gly.glyph.scale_mode = 'data_scaling_off'
        gly.glyph.color_mode = 'no_coloring'
        script.add_module(gly)
        
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
    if len(sys.argv)==3:
        mc.filename = sys.argv[1]
        try:
            mc.timestep = int(sys.argv[2])
        except ValueError:
            print "Timestep is not a number."
            sys.exit(1)
        if isfile(mc.filename):
            mc.main()
        else:
            sys.exit(1)
            print "File not found."
    else:
        print "[filename] [timestep]"