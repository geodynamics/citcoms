#! /usr/bin/env python
try:
    import wxversion
    wxversion.ensureMinimal('2.6')
except ImportError:
    pass

from enthought.mayavi.app import Mayavi
import sys
from os.path import isfile

class VtkDisplay(Mayavi):
 
    filename = None
        
    def run(self):

        from enthought.mayavi.sources.vtk_file_reader import VTKFileReader
        #import modules here
        from enthought.mayavi.modules import surface, glyph , axes, outline, orientation_axes, scalar_cut_plane  
        from enthought.mayavi.sources.vtk_data_source import VTKDataSource 
        from enthought.tvtk.api import tvtk
        #CitcomS filter
        from plugins.filter.CitcomSshowCaps import CitcomSshowCaps
        from plugins.filter.CitcomSreduce import CitcomSreduce
        import re
        
        
        script = self.script
        
        #DEFINES
        orange = (1.0,0.5,0)
                
        ################
        #Read Meta information
        meta = ""
        try:
            vtk = open(self.filename, "r")
            vtk.readline()
            meta = vtk.readline()
        except IOError:
            print 'cannot open file'
        try:
            print "Reading meta-information"
            m = re.search('(?<=NX:)\d+', meta)
            nx = int(m.group(0))
            print "NX: ", nx
            m = re.search('(?<=NY:)\d+', meta)
            ny = int(m.group(0))
            print "NY: ", ny
            m = re.search('(?<=NZ:)\d+', meta)
            nz = int(m.group(0))
            print "NZ: ", nz
            m = re.search('(?<=Radius_Inner:)(\d+|.)+', meta)
            print m.group(0)
            radius_inner = float(m.group(0))
            print "Radius Inner: ", radius_inner
            
        except ValueError:
            print "Non-valid meta information in file..."
    
        vtk.close()
        
        
        ################
        
        #Read Vtk file
        src_vtkf = VTKFileReader()
        src_vtkf.initialize(self.filename)
        
        ###########Display Data############
        #Create new scene
        script.new_scene()     
        
        
        script.add_source(src_vtkf)
        
        
        scap = CitcomSshowCaps()
        script.add_filter(scap)
        
        #Show ScalarCutPlane
        scp = scalar_cut_plane.ScalarCutPlane()
        script.add_module(scp)
        
        #Add filter for a reduce grid
        redu = CitcomSreduce()
        script.add_filter(redu)
       
        #Shows Glyph on reduce grid
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
    mc = VtkDisplay()
    #mc.filename = "/home/maweier/vtk_output_temp/general.0.vtk"
    mc.filename = sys.argv[1]
    if isfile(mc.filename):
        mc.main()
    else:
        print "Type filename of Vtkfile to display"
        sys.exit(1)