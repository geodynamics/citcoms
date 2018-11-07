"""A simple filter that thresholds CitcomS Caps from input data."""

# Author: Martin Weier
#
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
from enthought.traits import Instance, Range, Int
from enthought.traits.ui import View, Group, Item
from enthought.tvtk.api import tvtk

# Local imports
from enthought.mayavi.core.filter import Filter


######################################################################
# `Threshold` class.
######################################################################
class CitcomSshowCaps(Filter):

    # The version of this class.  Used for persistence.
    __version__ = 0

    # The threshold filter.
    ugrid_filter = Instance(tvtk.ExtractUnstructuredGrid, ())

    # Lower threshold (this is a dynamic trait that is changed when
    # input data changes).
    lower_threshold = Range(0, 12, 0,
                            desc='the lower threshold of the filter')

    # Upper threshold (this is a dynamic trait that is changed when
    # input data changes).
    upper_threshold = Range(0, 12, 12,
                            desc='the upper threshold of the filter')

    # Our view.
    view = View(Group(Item(name='lower_threshold'),
                      Item(name='upper_threshold'))
                )
    
   
    n = Int()
    caps = Int()
    
 
    
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
        self.ugrid_filter.point_clipping = 1
        self.ugrid_filter.merging = 0
        self.outputs = [self.ugrid_filter.output]
    
    def update_pipeline(self):
        """Override this method so that it *updates* the tvtk pipeline
        when data upstream is known to have changed.

        This method is invoked (automatically) when the input fires a
        `pipeline_changed` event.
        """
        # By default we set the input to the first output of the first
        # input.
        fil = self.ugrid_filter
        fil.input = self.inputs[0].outputs[0]
        
        #Than we calculate how many points belong to one cap
        self.caps = 12
        self.n = self.inputs[0].outputs[0].number_of_points/12
       
        #Than we set the output of the filter
        self.outputs[0] = fil.output
        self.outputs.append(self.inputs[0].outputs[0])
        self.pipeline_changed = True

    def update_data(self):
        """Override this method to do what is necessary when upstream
        data changes.

        This method is invoked (automatically) when any of the inputs
        sends a `data_changed` event.
        """
	fil = self.ugrid_filter
        fil.update()
        # Propagate the data_changed event.
        self.data_changed = True

    ######################################################################
    # Non-public interface
    ######################################################################
    def _lower_threshold_changed(self,old_value, new_value):
        """Callback interface for the lower threshold slider"""
        fil = self.ugrid_filter
        fil.point_minimum = (self.lower_threshold)*(self.n)
        fil.update()
        self.data_changed = True
        
        
    def _upper_threshold_changed(self, old_value, new_value):
        """Callback interface for the upper threshold slider"""
        fil = self.ugrid_filter
        fil.point_maximum = self.upper_threshold*(self.n)
        fil.update()
        self.data_changed = True
      
   
