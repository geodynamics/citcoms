"""A that shows a slice at a specified radius

"""
# Author: Martin Weier
# Copyright (c) 2006, California Institue of Technology
# License: GPL Style.

# Enthought library imports.
from enthought.traits import Instance, Range, Int, Float
from enthought.traits.ui import View, Group, Item
from enthought.tvtk import tvtk

# Local imports
from enthought.mayavi.core.filter import Filter


######################################################################
# `ShowSurface` class.
######################################################################
class ShowSurface(Filter):

    # The version of this class.  Used for persistence.
    __version__ = 0

    # The threshold filter.

    prog_filter = Instance(tvtk.ProgrammableFilter, ())

    
    # Upper threshold (this is a dynamic trait that is changed when
    # input data changes).
    surfacelevel = Range(1, 17, 1,
                            desc='the surface filter')

    # Our view.
    view = View(Group(Item(name='surfacelevel')
                ))
    
    current_level=Int()
    nx = Int()
    ny = Int()
    nz = Int()
    
    
    
    
    def setvalues(self,nx,ny,nz,level):
        """This Method needs to be set before the execution of the filter
        it accepts nx,ny,nz,level"""
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.current_level = level
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
        self.prog_filter.set_execute_method(self._showsurface) 
        self.outputs = [self.prog_filter.output]
    
    def update_pipeline(self):
        """Override this method so that it *updates* the tvtk pipeline
        when data upstream is known to have changed.

        This method is invoked (automatically) when the input fires a
        `pipeline_changed` event.
        """
        # By default we set the input to the first output of the first
        # input.
        fil = self.prog_filter
        fil.input = self.inputs[0].outputs[0]

        # We force the ranges to be reset to the limits of the data.
        # This is because if the data has changed upstream, then the
        # limits of the data must be changed.
        #self._update_ranges(reset=True)
        
        #fil.threshold_between(self.lower_threshold, self.upper_threshold)
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

       
        # Propagate the data_changed event.
        self.prog_filter.set_execute_method(self._showsurface) 
        self.outputs = [self.prog_filter.output]
        self.data_changed = True

    
    def _showsurface(self):
        print "showsurface update"
        print self.current_level
        input = self.prog_filter.unstructured_grid_input
        numCells = input.number_of_cells
        quadgrid = tvtk.UnstructuredGrid()
        quadgrid.allocate(1,1)
    
        reduced_points = []
        reduced_scalars = []
        reduced_vectors = []
        j = 1
        cell_count=0
        for i in xrange(numCells):
            if j==self.current_level:
                cell = input.get_cell(i)
                scalars = input.point_data.scalars
                vectors = input.point_data.vectors
                point_ids = cell.point_ids
                points = cell.points
                reduced_points.append(points[2])
                reduced_points.append(points[1])
                reduced_points.append(points[5])
                reduced_points.append(points[6])
           
                reduced_scalars.append(scalars[point_ids[2]])
                reduced_scalars.append(scalars[point_ids[1]])
                reduced_scalars.append(scalars[point_ids[5]])
                reduced_scalars.append(scalars[point_ids[6]])
            
            
                reduced_vectors.append(vectors[point_ids[2]]) 
                reduced_vectors.append(vectors[point_ids[1]])
                reduced_vectors.append(vectors[point_ids[5]])
                reduced_vectors.append(vectors[point_ids[6]])
            
                quadgrid.insert_next_cell(9,[cell_count,cell_count+1,cell_count+2,cell_count+3])
                cell_count+=4
            
            if j == self.nx:
                j=1
    
            j+=1
        
        vtkReduced_vectors = tvtk.FloatArray()
        vtkReduced_scalars = tvtk.FloatArray()
        vtkReduced_vectors.from_array(reduced_vectors)
        vtkReduced_scalars.from_array(reduced_scalars)
    
        vtkReduced_scalars.name = 'Scalars'
        vtkReduced_vectors.name = 'Vectors'
    
        #showsurfF.unstructured_grid_output = quadgrid
        self.prog_filter.unstructured_grid_output.set_cells(9,quadgrid.get_cells())
        self.prog_filter.unstructured_grid_output.point_data.scalars = vtkReduced_scalars
        self.prog_filter.unstructured_grid_output.point_data.vectors = vtkReduced_vectors
        self.prog_filter.unstructured_grid_output.points = reduced_points  
    
        
    ######################################################################
    # Non-public interface
    ######################################################################
    def _surfacelevel_changed(self, new_value):
        fil = self.prog_filter
        print self.current_level
        self.current_level = new_value-1
        self._showsurface()
        fil.update()
        self.data_changed = True
    
    
