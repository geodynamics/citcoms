"""A simple filter that thresholds on input data.

"""
# Author: Prabhu Ramachandran <prabhu_r@users.sf.net>
# Copyright (c) 2005, Enthought, Inc.
# License: BSD Style.

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
    lower_threshold = Range(0, 13, 0,
                            desc='the lower threshold of the filter')

    # Upper threshold (this is a dynamic trait that is changed when
    # input data changes).
    upper_threshold = Range(0, 13, 13,
                            desc='the upper threshold of the filter')

    # Our view.
    view = View(Group(Item(name='lower_threshold'),
                      Item(name='upper_threshold'))
                )
    
    nx = Int()
    ny = Int()
    nz = Int()
    n = Int()
    caps = Int()
    
    def setvalues(self,nxin,nyin,nzin):
        self.nx = nxin
        self.ny = nyin
        self.nz = nzin
    
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
        self.caps = 12
        self.n = self.inputs[0].outputs[0].number_of_points/12
        #self.inputs[0].outputs[0].number_of_points/(self.nx*self.ny*self.nz)
        # We force the ranges to be reset to the limits of the data.
        # This is because if the data has changed upstream, then the
        # limits of the data must be changed.
        #self._update_ranges(reset=True)
        
        #fil.threshold_between(self.lower_threshold, self.upper_threshold)
        #fil.update()
        self.outputs[0] = fil.output
        self.outputs.append(self.inputs[0].outputs[0])
        self.pipeline_changed = True

    def update_data(self):
        """Override this method to do what is necessary when upstream
        data changes.

        This method is invoked (automatically) when any of the inputs
        sends a `data_changed` event.
        """

        #self._update_ranges(reset=True)

        # XXX: These are commented since updating the ranges does
        # everything for us.
        
        #fil = self.threshold_filter
        #fil.threshold_between(self.lower_threshold, self.upper_threshold)
        #fil.update()
        # Propagate the data_changed event.
        self.data_changed = True

    ######################################################################
    # Non-public interface
    ######################################################################
    def _lower_threshold_changed(self,old_value, new_value):
        #if new_value <> self.upper_threshold and new_value<upper_threshold:
        fil = self.ugrid_filter
        fil.point_minimum = (self.lower_threshold)*(self.n)
        fil.update()
        self.data_changed = True
        #else:                            #Create a single point to prevent that other filters crash because of missing input
        #    self.outputs[0].points = [(100,100,100)]
        
    def _upper_threshold_changed(self, old_value, new_value):
        #if new_value <> self.lower_threshold and new_value>lower_threshold:
        fil = self.ugrid_filter
        fil.point_maximum = self.upper_threshold*(self.n)
        fil.update()
        self.data_changed = True
        #else:
        #self.outputs[0].points = [(100,100,100)]
        

   
