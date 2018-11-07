"""A filter that reduces CitcomS vtk input data.
"""

#Author: Martin Weier 
#Copyright (C) 2006  California Institute of Technology
#This program is free software; you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation; either version 2 of the License, or
#any later version.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#You should have received a copy of the GNU General Public License
#along with this program; if not, write to the Free Software
#Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA


# Enthought library imports.
from enthought.traits import Instance, Range, Int, Float, Enum, Trait, Button
from enthought.traits.ui import View, Group, Item
from enthought.tvtk.api import tvtk

# Local imports
from enthought.mayavi.core.filter import Filter

from CitcomSSphere import *

######################################################################
# `Threshold` class.
######################################################################
class CitcomSreduce(Filter):

    # The version of this class.  Used for persistence.
    __version__ = 0

    # The threshold filter.

    probe_filter = Instance(tvtk.ProbeFilter, ())

    citcomsgrid = CitcomSSphere()
    sphere = Instance(tvtk.SphereSource,())
    
    # Upper threshold (this is a dynamic trait that is changed when
    # input data changes).
    Radius = Range(0.0, 1.0, 0.0, desc='adjust radius')
    
    theta = Range(3, 40, 3, desc='the theta resolution')
    phi = Range(3, 40, 3, desc='the upper threshold of the filter')

    Selected_Source = Enum(  'Sphere','CitcomSGrid',)
    
    radius_max = Float(1.0)
    set_radius_max = Button('Set Radius Max')
    # Our view.
    view = View(Item(name="Selected_Source"),
                Group(Item(name='Radius'),
                      Item(name='theta'),
                      Item(name='phi'),
                      Item(name='radius_max'),
                      Item(name='set_radius_max', style='simple', label='Simple'),
                      show_border = True
                      ),
                )
    
    grid_source = True
    
    ######################################################################
    # `Filter` interface.
    ######################################################################
    
    def setup_pipeline(self):
        """Override this method so that it *creates* its tvtk
        pipeline.

        This method is invoked when the object is initialized via
        `__init__`.  Note that at the time this method is called, the
        tvtk data pipeline will *not* yet be setup.  So upstream data
        will not be available.  The idea is that you simply create the
        basic objects and setup those parts of the pipeline not
        dependent on upstream sources and filters.
        """
        # Just setup the default output of this filter.
        s = self.sphere
        s.set(radius=0.0,theta_resolution=20,phi_resolution=20)
        self.probe_filter.input = s.output
        self.outputs = [self.probe_filter.output]
    
    def update_pipeline(self):
        """Override this method so that it *updates* the tvtk pipeline
        when data upstream is known to have changed.

        This method is invoked (automatically) when the input fires a
        `pipeline_changed` event.
        """
        # By default we set the input to the first output of the first
        # input.
        fil = self.probe_filter
        
        fil.source = self.inputs[0].outputs[0]
    
        
        self._calc_grid(0,self.theta,self.phi)
        fil.update()
    
        self.outputs[0] = fil.output
        self.pipeline_changed = True

    def update_data(self):
        """Override this method to do what is necessary when upstream
        data changes.

        This method is invoked (automatically) when any of the inputs
        sends a `data_changed` event.
        """

        self.probe_filter.source = self.inputs[0].outputs[0]
        self.probe_filter.update()
        # Propagate the data_changed event.
        self.data_changed = True


    def _calc_grid(self,radius,resolution_x,resolution_y):
        
        fil = self.probe_filter
       
        coords = []
        
        if self.Selected_Source == 'CitcomSGrid':
           
            for i in xrange(12):
                          
                coords += self.citcomsgrid.coords_of_cap(radius,self.theta,self.phi,i)
            
            grid = tvtk.UnstructuredGrid()
            
            #Connectivity for 2d-Data
            #There is no need to interpolate with the CitcomS grid surface. If this is however
            #wanted uncomment this code to create the CitcomS surface information
            #for capnr in xrange(12):
            #    i=1
            #    for n in xrange((resolution_x+1)*(resolution_y+1) - (resolution_x+1)):
            #        if i%(resolution_x+1)!=0 :
            #            n0 = n+(capnr*((resolution_x+1)*(resolution_y+1)))
            #            n1 = n0+1
            #            n2 = n0+resolution_y+1
            #            n3 = n2+1          
            #            grid.insert_next_cell(8,[n0,n1,n2,n3])
            #        i+=1
                
            ##        
            
            grid.points = coords
            fil.input = grid
                
        if self.Selected_Source == 'Sphere':
             sphere = tvtk.SphereSource()
             sphere.radius = radius
             sphere.theta_resolution = resolution_x
             sphere.phi_resolution = resolution_y
             
             #Rotate the Sphere so that the poles are at the right location
             transL = tvtk.Transform()
             trans1 = tvtk.TransformPolyDataFilter()
             trans2 = tvtk.TransformPolyDataFilter()
             trans1.input = sphere.output
             transL.rotate_y(90)
             transL.update()
             trans1.transform = transL
             trans1.update()
             trans2.input = trans1.output
             transL.rotate_z(90)
             transL.update()
             trans2.transform = transL
             trans2.update()
             
             fil.input = trans2.output
             
        
       
        fil.update()


    ######################################################################
    # Non-public interface
    ######################################################################
    def _Radius_changed(self, new_value):
        fil = self.probe_filter
        #self.sphere.radius = new_value
        self._calc_grid(new_value,self.theta,self.phi)
        fil.update()
        self.data_changed = True
        

    def _theta_changed(self, new_value):
        fil = self.probe_filter
        self._calc_grid(self.Radius,new_value,self.phi)
        fil.update()
        self.data_changed = True
     
    
    def _phi_changed(self, new_value):
        fil = self.probe_filter
        self._calc_grid(self.Radius,self.theta,new_value)
        fil.update()
        self.data_changed = True
        
    def _Selected_Source_changed(self,new_value):
        self._calc_grid(self.Radius, self.theta, self.phi)
        self.outputs[0] = self.probe_filter.output
        self.data_changed = True
        self.pipeline_changed = True
        
    def _radius_max_changed(self,new_value):
        if self.Radius > new_value:
            self.Radius = new_value
        if new_value <= 0.0:
            self.radius_max = 0.0
 
    def _set_radius_max_fired(self):
        trait = Range(0.0, self.radius_max, self.Radius,
                          desc='adjust radius')
        self.add_trait('Radius', trait)
        
