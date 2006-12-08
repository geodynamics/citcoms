"""This source manages a CitcomS Hdf file given to it.  
"""
# Author: Martin Weier
# Copyright (c) 2006, California Institute of Technology
# License: GPL Style

# Enthought library imports.
from enthought.traits import Trait, TraitPrefixList, Instance, Int, Str, Button
from enthought.traits.ui import View, Group, Item
from enthought.traits.ui.menu import OKButton
from enthought.persistence.state_pickler \
     import gzip_string, gunzip_string, set_state
from enthought.tvtk.api import tvtk

from citcoms_plugins.utils import parsemodel
from CitcomSHDFUgrid import CitcomSHDFUgrid

# Local imports.
from enthought.mayavi.core.source import Source
from enthought.mayavi.core.common import handle_children_state
from enthought.mayavi.sources.vtk_xml_file_reader import get_all_attributes
from CitcomHDFThread import CitcomSHdf2UGridThread

import tables
import re


######################################################################
# `CitcomSHDFFileReader` class
######################################################################
class CitcomSHDFFileReader(Source):

    """This source manages a CitcomS Hdf file given to it. """

    # The version of this class.  Used for persistence.
    __version__ = 0

    # The VTK dataset to manage.
    data = Instance(tvtk.DataSet)
    # Class to convert Hdf to Vtk Unstructured Grid Objects
    citcomshdftougrid = CitcomSHDFUgrid()

    timestep = Int(0)

    #To support changing the Scalar values in Mayavi2
    temperature = Instance(tvtk.FloatArray())
    viscosity = Instance(tvtk.FloatArray())
    
    #Resolution comming from Hdf file
    nx = Int()
    ny = Int()
    nz = Int()
    
    #Current reduced resolution. User defined
    nx_redu = Int()
    ny_redu = Int()
    nz_redu = Int()
    
    #Number of timesteps in Hdf
    #timesteps = Int()
    
    #Button to trigger the process of reading a timestep
    read_data = Button('Read data')
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
    view = View(Group(#Item(name='current_timestep'),
                      Item(name='nx_redu', label='Grid Size in X'),
                      Item(name='ny_redu', label='Grid Size in Y'),
                      Item(name='nz_redu', label='Grid Size in Z'),
                      Item(name='read_data', style='simple', label='Simple'),
                      Item(name='point_scalars_name'),
                      Item(name='point_vectors_name'),
                      Item(name='point_tensors_name'),
                      #Item(name='cell_scalars_name'),
                      #Item(name='cell_vectors_name'),
                      #Item(name='cell_tensors_name'),
                     ),
               )
    
    ######################################################################
    # `object` interface
    ######################################################################
    #Invoked during process of saving the visualization to a file
    def __get_pure_state__(self):
        d = super(CitcomSHDFFileReader, self).__get_pure_state__()
        output = "Filename: " +self.filename+ "NX:%d NY:%d NZ:%d" %(self.nx_redu,self.ny_redu,self.nz_redu)
        z = gzip_string(output)
        d['data'] = z
        return d
    
    #When the Visualisation is opened again this Method is called
    def __set_pure_state__(self, state):
        z = state.data
        if z:
            d = gunzip_string(z)
            m = re.search('(?<=Filename:)\w+', header)
            file_name = m.group(0)
            m = re.search('(?<=NX:)\w+', d)
            self.nx_redu = int(m.group(0))
            m = re.search('(?<=NX:)\w+', d)
            self.ny_redu = int(m.group(0))
            m = re.search('(?<=NX:)\w+', d)
            self.nz_redu = int(m.group(0))
        
            if not isfile(file_name):
                msg = 'Could not find file at %s\n'%file_name
                msg += 'Please move the file there and try again.'
                raise IOError, msg
            
            (step, modelname, metafilepath, filepath) = parsemodel(file_name)
            file_name = metafilepath
            
            self.filename = file_name
        
            #self.data = self.citcomshdftougrid.initialize(self.filename,0,0,0,0,False,False)
            f = tables.openFile(file_name,'r')
            self.nx = int(f.root.input._v_attrs.nodex)
            self.ny = int(f.root.input._v_attrs.nodey)
            self.nz = int(f.root.input._v_attrs.nodez)
        
            #self.timesteps = int(f.root.time.nrows)

            f.close()
            self.data = self.citcomshdftougrid.initialize(self.filename,self.current_timestep,self.nx_redu,self.ny_redu,self.nz_redu)
            self.data_changed = True
            self._update_data()
            # Now set the remaining state without touching the children.
            set_state(self, state, ignore=['children', '_file_path'])
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
        """This Methods initializes the reader and reads the meta-information from the Hdf file """
        self.filename = file_name
        
        #self.data = self.citcomshdftougrid.initialize(self.filename,0,0,0,0,False,False)
        from citcoms_plugins.utils import parsemodel

        (step, modelname, coordpath, filepath) = parsemodel(file_name)
        if step is None:
            step = 0

        f = tables.openFile(coordpath,'r')
        self.nx = int(f.root.input._v_attrs.nodex)
        self.ny = int(f.root.input._v_attrs.nodey)
        self.nz = int(f.root.input._v_attrs.nodez)
        
        self.nx_redu = self.nx
        self.ny_redu = self.ny
        self.nz_redu = self.nz
        
        self.timestep = step
        #self.timesteps = int(f.root.time.nrows)
        f.close()
        
        
    ######################################################################
    # `TreeNodeObject` interface
    ######################################################################
    def tno_get_label(self, node):
        """ Gets the label to display for a specified object.
        """
        ret = "CitcomS HDF5 Data (not initialized)"
        if self.data:
            typ = self.data.__class__.__name__
            ret = "CitcomS HDF5 Data (step %d)" % self.timestep
        return ret

    ######################################################################
    # Non-public interface
    ######################################################################
    def _data_changed(self, data):
        """Invoked when the upsteam data sends an data_changed event"""
        self._update_data()
        self.outputs = [data]
        self.data_changed = True
        # Fire an event so that our label on the tree is updated.
        self.trait_property_changed('name', '',
                                    self.tno_get_label(None))
        
    ##Callbacks for our traits    
    def _current_timestep_changed(self,new_value):
        """Callback for the current timestep input box"""
        if new_value < 0:
            self.current_timestep = 0
        #if new_value > self.timesteps:
        #    self.current_timestep = self.timesteps-1
    
    def _read_data_fired(self):
        """Callback for the Button to read the data"""
        
        
        self.data = self.citcomshdftougrid.initialize(self.filename,self.timestep,self.nx_redu,self.ny_redu,self.nz_redu)
        self.temperature = self.citcomshdftougrid.get_vtk_temperature()
        self.viscosity = self.citcomshdftougrid.get_vtk_viscosity()
        
        ##New Thread Code
        #thread1 = CitcomSHdf2UGridThread()
        #thread2 = CitcomSProgressBar()
        
        #thread1.set_citcomsreader(self.filename,self.current_timestep,self.nx_redu,self.ny_redu,self.nz_redu,self.thread_callback)
        #progress = thread1.get_ref()
        #thread1.start()
        #thread2.set_ref(progress)
        #thread2.start()
        
        self.data_changed = True
        self._update_data()
        
    def _nx_redu_changed(self, new_value):
        """callback for the nx_redu input box"""
        if new_value < 1:
            self.nx_redu = 1
        if new_value > self.nx:
            self.nx_redu = self.nx
                     
    def _ny_redu_changed(self, new_value):
        """callback for the ny_redu input box"""
        if new_value < 1:
            self.ny_redu = 1
        if new_value > self.ny:
            self.ny_redu = self.ny
        
    
    def _nz_redu_changed(self, new_value):
        """callback for the nz_redu input box"""
        if new_value < 1:
            self.nz_redu = 1
        if new_value > self.nz:
            self.nz_redu = self.nz
  
    def _point_scalars_name_changed(self, value):
        if value == "Temperature":
            self.data.point_data.scalars = self.temperature
        if value == "Viscosity":
            self.data.point_data.scalars = self.viscosity
        self.data_changed = True
        self._set_data_name('scalars', 'point', value)

    def _point_vectors_name_changed(self, value):
        self._set_data_name('vectors', 'point', value)

    def _point_tensors_name_changed(self, value):
        self._set_data_name('tensors', 'point', value)


    ########################Non Public##############################
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

        
    def _update_data(self):
        if not self.data:
            return
        else:
            trait = Trait('Temperature', TraitPrefixList('Temperature','Viscosity'))
            self.add_trait('point_scalars_name', trait)
            trait = Trait('Velocity', TraitPrefixList('Velocity'))
            self.add_trait('point_vectors_name', trait)
            
    def thread_callback(self,hexagrid,vtk_viscosity,vtk_temperature):
        hexagrid.print_traits()
        self.data = hexagrid
        self.temperature = vtk_temperature
        self.viscosity = vtk_temperature
        
        
        
