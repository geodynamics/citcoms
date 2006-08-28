"""A filter that reduces CitcomS vtk input data.
"""

# Author: Martin Weier 
# Copyright (c) 2006, California Institute of Technology


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
    
    # Upper threshold (this is a dynamic trait that is changed when
    # input data changes).
    Radius = Range(0.0, 1.0, 0.80,
                            desc='adjust radius')
    
    theta = Range(0, 40, 10,
                            desc='the theta resolution')
    phi = Range(0, 40, 10,
                            desc='the upper threshold of the filter')

    Selected_Source = Enum( 'Sphere', 'CitcomSGrid',)
    
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
    def setvalues(self,nx,ny,nz):
        pass
    
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
        
        #Filter needs to be connected to the unfiltered Dataset.
        #If the filter above has changed its data connect to output port 1 instead of 2
        fil.source = self.inputs[0].outputs[0]
    
        #self.sphere.radius = 0.98 
        #self.sphere.theta_resolution = 24 
        #self.sphere.phi_resolution = 24
        #fil.input = self.sphere.output  
        
        self._calc_grid(0,self.theta,self.phi)
        fil.update()
        # We force the ranges to be reset to the limits of the data.
        # This is because if the data has changed upstream, then the
        # limits of the data must be changed.
        #self._update_ranges(reset=True)
        #fil.update()
        self.outputs[0] = fil.output
        self.pipeline_changed = True

    def update_data(self):
        """Override this method to do what is necessary when upstream
        data changes.

        This method is invoked (automatically) when any of the inputs
        sends a `data_changed` event.
        """

        #self._update_ranges(reset=True)
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
            grid.points = coords
            #dl = tvtk.Delaunay3D()
            #dl.input = grid
            #fil.input = dl.output
            fil.input = grid
                
        if self.Selected_Source == 'Sphere':
             sphere = tvtk.SphereSource()
             sphere.radius = radius
             sphere.theta_resolution = resolution_x
             sphere.phi_resolution = resolution_y
             fil.input = sphere.output  
             
        
       
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
        self._calc_grid(self.Radius,self.phi,new_value)
        fil.update()
        self.data_changed = True
        
    def _Selected_Source_changed(self,new_value):
        self._calc_grid(self.Radius, self.theta, self.phi)
        self.outputs[0] = self.probe_filter.output
        self.data_changed = True
        self.pipeline_changed = True
        
    def _radius_max_changed(self,new_value):
        if l > new_value:
            self.Radius = new_value
        if new_value <= 0.0:
            self.radius_max = 0.0
 
    def _set_radius_max_fired(self):
        trait = Range(0.0, new_value, self.Radius,
                          desc='adjust radius')
        self.add_trait('Radius', trait)
        
        
    def _update_ranges(self, reset=False):
        pass
