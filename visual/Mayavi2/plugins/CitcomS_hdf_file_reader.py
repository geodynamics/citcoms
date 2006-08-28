"""This source manages a CitcomS Hdf file given to it.  
"""
# Author: Martin Weier
# Copyright (c) 

# Enthought library imports.
from enthought.traits import Trait, TraitPrefixList, Instance, Int, Str, Button
from enthought.traits.ui import View, Group, Item
from enthought.traits.ui.menu import OKButton
from enthought.persistence.state_pickler \
     import gzip_string, gunzip_string, set_state
from enthought.tvtk.api import tvtk
from enthought.mayavi.plugins.CitcomSHDFUgrid import CitcomSHDFUgrid

# Local imports.
from enthought.mayavi.core.source import Source
from enthought.mayavi.core.common import handle_children_state
from enthought.mayavi.sources.vtk_xml_file_reader import get_all_attributes
import tables

######################################################################
# `CitcomSVTKDataSource` class
######################################################################
class CitcomSHDFFileReader(Source):

    """This source manages a CitcomS Hdf file given to it. """

    # The version of this class.  Used for persistence.
    __version__ = 0

    # The VTK dataset to manage.
    data = Instance(tvtk.DataSet)
    citcomshdftougrid = CitcomSHDFUgrid()
    current_timestep = Int(0)
    nx = Int()
    ny = Int()
    nz = Int()
    
    nx_redu = Int()
    ny_redu = Int()
    nz_redu = Int()
    
    timesteps = Int()
    frequency = Int()
    read_timestep = Button('Read timestep')
    filename = Str()
    ########################################
    # Dynamic traits: These traits are dummies and are dynamically
    # updated depending on the contents of the file.

    # The active point scalar name.
    point_scalars_name = Trait('', TraitPrefixList(['']))
    # The active point vector name.
    point_vectors_name = Trait('', TraitPrefixList(['']))
    # The active point tensor name.
    point_tensors_name = Trait('', TraitPrefixList(['']))

    # The active cell scalar name.
    cell_scalars_name = Trait('', TraitPrefixList(['']))
    # The active cell vector name.
    cell_vectors_name = Trait('', TraitPrefixList(['']))
    # The active cell tensor name.
    cell_tensors_name = Trait('', TraitPrefixList(['']))
    ########################################

    # Our view.
    view = View(Group(Item(name='current_timestep'),
                      Item(name='nx_redu'),
                      Item(name='ny_redu'),
                      Item(name='nz_redu'),
                      Item(name='read_timestep', style='simple', label='Simple'),
                      Item(name='point_scalars_name'),
                      Item(name='point_vectors_name'),
                      Item(name='point_tensors_name'),
                      Item(name='cell_scalars_name'),
                      Item(name='cell_vectors_name'),
                      Item(name='cell_tensors_name'),
                      
                      ),
                      
                     )
    
    ######################################################################
    # `object` interface
    ######################################################################
    def __get_pure_state__(self):
        d = super(VTKDataSource, self).__get_pure_state__()
        data = self.data
        if data:
            w = tvtk.DataSetWriter(write_to_output_string=1)
            warn = w.global_warning_display
            w.set_input(data)
            w.global_warning_display = 0
            w.update()
            w.global_warning_display = warn
            z = gzip_string(w.output_string)
            d['data'] = z
        return d

    def __set_pure_state__(self, state):
        z = state.data
        if z:
            d = gunzip_string(z)
            r = tvtk.DataSetReader(read_from_input_string=1,
                                   input_string=d)
            r.update()
            self.data = r.output
        # Now set the remaining state without touching the children.
        set_state(self, state, ignore=['children', 'data'])
        # Setup the children.
        handle_children_state(self.children, state.children)
        # Setup the children's state.
        set_state(self, state, first=['children'], ignore=['*'])

    ######################################################################
    # `Base` interface
    ######################################################################
    def start(self):
        """This is invoked when this object is added to the mayavi
        pipeline.
        """
        # Do nothing if we are already running.
        if self.running:
            return

        # Update the data just in case.
        self._update_data()

        # Call the parent method to do its thing.  This will typically
        # start all our children.
        super(CitcomSHDFFileReader, self).start()

    def update(self):
        """Invoke this to flush data changes downstream."""
        self.data_changed = True

    
    def initialize(self,file_name):
        self.filename = file_name
        
        #self.data = self.citcomshdftougrid.initialize(self.filename,0,0,0,0,False,False)
        f = tables.openFile(file_name,'r')
        self.nx = int(f.root.input._v_attrs.nodex)
        self.ny = int(f.root.input._v_attrs.nodey)
        self.nz = int(f.root.input._v_attrs.nodez)
        
        self.nx_redu = self.nx
        self.ny_redu = self.ny
        self.nz_redu = self.nz
        
        self.timesteps =  int(f.root.input._v_attrs.steps)
        self.frequency =  int(f.root.input._v_attrs.monitoringFrequency)
        
    ######################################################################
    # `TreeNodeObject` interface
    ######################################################################
    def tno_get_label(self, node):
        """ Gets the label to display for a specified object.
        """
        ret = "CitcomS HDF Data (uninitialized)"
        if self.data:
            typ = self.data.__class__.__name__
            ret = "CitcomS HDF Data (%d)"%self.current_timestep
        return ret

    ######################################################################
    # Non-public interface
    ######################################################################
    def _data_changed(self, data):
        self._update_data()
        self.outputs = [data]
        self.data_changed = True
        # Fire an event so that our label on the tree is updated.
        self.trait_property_changed('name', '',
                                    self.tno_get_label(None))
        
    def _current_timestep_changed(self,new_value):
        if new_value < 0:
            current_timestep = 0
        if new_value > self.timesteps:
            current_timestep = 100
    
    def _read_timestep_fired(self):
        self.data = self.citcomshdftougrid.initialize(self.filename,self.current_timestep,self.nx_redu,self.ny_redu,self.nz_redu,False,False)
        
    def _nx_redu_changed(self, new_value):
        if new_value < 1:
            self.nx_redu = 1
        if new_value > self.nx:
            self.nx_redu = self.nx
                     
    def _ny_redu_changed(self, new_value):
        if new_value < 1:
            self.ny_redu = 1
        if new_value > self.ny:
            self.ny_redu = self.ny
        
    
    def _nz_redu_changed(self, new_value):
        if new_value < 1:
            self.nz_redu = 1
        if new_value > self.nz:
            self.nz_redu = self.nz
  
    def _set_data_name(self, data_type, attr_type, value):
        if not value:
            return
        dataset = self.data
        data = None
        if attr_type == 'point':
            data = dataset.point_data
        elif attr_type == 'cell':
            data = dataset.cell_data
        meth = getattr(data, 'set_active_%s'%data_type)
        meth(value)
        self.update()
        # Fire an event, so the changes propagate.
        self.data_changed = True

    def _point_scalars_name_changed(self, value):
        self._set_data_name('scalars', 'point', value)

    def _point_vectors_name_changed(self, value):
        self._set_data_name('vectors', 'point', value)

    def _point_tensors_name_changed(self, value):
        self._set_data_name('tensors', 'point', value)

    def _cell_scalars_name_changed(self, value):
        self._set_data_name('scalars', 'cell', value)

    def _cell_vectors_name_changed(self, value):
        self._set_data_name('vectors', 'cell', value)

    def _cell_tensors_name_changed(self, value):
        self._set_data_name('tensors', 'cell', value)
    
    def _update_data(self):
        if not self.data:
            return
        pnt_attr, cell_attr = get_all_attributes(self.data)
        
        def _setup_data_traits(obj, attributes, d_type):
            attrs = ['scalars', 'vectors', 'tensors']
            data = getattr(obj.data, '%s_data'%d_type)
            for attr in attrs:
                values = attributes[attr]
                if values:
                    default = getattr(obj, '%s_%s_name'%(d_type, attr))
                    if default and default in values:
                        pass
                    else:
                        default = values[0]
                    trait = Trait(default, TraitPrefixList(values))
                    getattr(data, 'set_active_%s'%attr)(default)
                else:
                    trait = Trait('', TraitPrefixList(['']))
                obj.add_trait('%s_%s_name'%(d_type, attr), trait)
        
        _setup_data_traits(self, pnt_attr, 'point')
        _setup_data_traits(self, cell_attr, 'cell')
        
