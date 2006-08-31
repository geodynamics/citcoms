"""A VTK file reader object.

"""
# Author: Prabhu Ramachandran <prabhu_r@users.sf.net>
# Copyright (c) 2005, Enthought, Inc.
# License: BSD Style.


# Standard library imports.
import re
from os.path import split, join, isfile
from glob import glob
from os.path import basename

# Enthought library imports.
from enthought.traits import Trait, TraitPrefixList,List, Str, Range, Instance, Int, Float
from enthought.traits.ui import View, Group, Item, Include
from enthought.persistence.state_pickler import set_state
from enthought.persistence.file_path import FilePath
from enthought.tvtk.api import tvtk

# Local imports
from enthought.mayavi.core.source import Source
from enthought.mayavi.core.common import handle_children_state


######################################################################
# Utility functions.
######################################################################
def get_file_list(file_name):
    """ Given a file name, this function treats the file as a part of
    a series of files based on the index of the file and tries to
    determine the list of files in the series.  The file name of a
    file in a time series must be of the form 'some_name[0-9]*.ext'.
    That is the integers at the end of the file determine what part of
    the time series the file belongs to.  The files are then sorted as
    per this index."""

    # The matching is done only for the basename of the file.
    f_dir, f_base = split(file_name)
    # Find the head and tail of the file pattern.
    head = re.sub("[0-9]+[^0-9]*$", "", f_base)
    tail = re.sub("^.*[0-9]+", "", f_base)
    pattern = head+"[0-9]*"+tail
    # Glob the files for the pattern.
    _files = glob(join(f_dir, pattern))

    # A simple function to get the index from the file.
    def _get_index(f, head=head, tail=tail):
        base = split(f)[1]
        result = base.replace(head, '')
        return float(result.replace(tail, ''))
        
    # Before sorting make sure the files in the globbed series are
    # really part of a timeseries.  This can happen in cases like so:
    # 5_2_1.vtk and 5_2_1s.vtk will be globbed but 5_2_1s.vtk is
    # obviously not a valid time series file.
    files = []
    for x in _files:
        try:
            _get_index(x)
        except ValueError:
            pass
        else:
            files.append(x)
        
    # Sort the globbed files based on the index value.
    def file_sort(x, y):
        x1 = _get_index(x)
        y1 = _get_index(y)
        if x1 > y1:
            return 1
        elif y1 > x1:
            return -1
        else:
            return 0

    files.sort(file_sort)
    return files


class CitcomSVTKFileReader(Source):

    """A CitcomS VTK file reader.  
    """

    # The version of this class.  Used for persistence.
    __version__ = 0
    nx = Int()
    ny = Int()
    nz = Int()
    radius_inner = Float()
    
    # The list of file names for the timeseries.
    file_list = List(Str, desc='a list of files belonging to a time series')

    # The current time step (starts with 0).  This trait is a dummy
    # and is dynamically changed when the `file_list` trait changes.
    # This is done so the timestep bounds are linked to the number of
    # the files in the file list.
    timestep = Range(0, 0, desc='the current time step')

    # A timestep view group that may be included by subclasses.
    time_step_group = Group(Item(name='_file_path', style='readonly'),
                            Item(name='timestep',
                                 defined_when='len(object.file_list) > 1')
                            )
    
    ##################################################
    # Private traits.
    ##################################################    

    # The current file name.  This is not meant to be touched by the
    # user.    
    _file_path = Instance(FilePath, (), desc='the current file name')


    ########################################
    # Dynamic traits: These traits are dummies and are dynamically
    # updated depending on the contents of the file.

    # The active scalar name.
    scalars_name = Trait('', TraitPrefixList(['']))
    # The active vector name.
    vectors_name = Trait('', TraitPrefixList(['']))
    # The active tensor name.
    tensors_name = Trait('', TraitPrefixList(['']))

    # The active normals name.
    normals_name = Trait('', TraitPrefixList(['']))
    # The active tcoord name.
    t_coords_name = Trait('', TraitPrefixList(['']))
    # The active field_data name.
    field_data_name = Trait('', TraitPrefixList(['']))
    ########################################

    # The VTK data file reader.
    reader = Instance(tvtk.DataSetReader, ())    

    # Our view.
    view = View(Group(Include('time_step_group'),
                      Item(name='scalars_name'),
                      Item(name='vectors_name'),
                      Item(name='tensors_name'),
                      Item(name='normals_name'),
                      Item(name='t_coords_name'),
                      Item(name='field_data_name'),
                      Item(name='reader'),
                      ))

    ######################################################################
    # `object` interface
    ######################################################################
    ######################################################################
    # `object` interface
    ######################################################################
    def __get_pure_state__(self):
        d = super(FileDataSource, self).__get_pure_state__()
        # These are obtained dynamically, so don't pickle them.
        for x in ['file_list', 'timestep']:
            d.pop(x, None)
        return d
    
    def __set_pure_state__(self, state):
        # The reader has its own file_name which needs to be fixed.
        state.reader.file_name = state._file_path.abs_pth
        # Now call the parent class to setup everything.
        # Use the saved path to initialize the file_list and timestep.
        fname = state._file_path.abs_pth
        if not isfile(fname):
            msg = 'Could not find file at %s\n'%fname
            msg += 'Please move the file there and try again.'
            raise IOError, msg
        
        self.initialize(fname)
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
        self.update_data()
        self.update()

        # Call the parent method to do its thing.  This will typically
        # start all our children.
        super(CitcomSVTKFileReader, self).start()

    def stop(self):
        """Invoked when this object is removed from the mayavi
        pipeline.
        """
        if not self.running:
            return

        # Call the parent method to do its thing.
        super(CitcomSVTKFileReader, self).stop()
    
    
    ######################################################################
    # `FileDataSource` interface
    ######################################################################
    def update(self):
        if not self._file_path.get():
            return
        reader = self.reader
        reader.update()
        self.render()

    def update_data(self):
        if not self._file_path.get():
            return
        attrs = ['scalars', 'vectors', 'tensors', 'normals',
                 't_coords', 'field_data']
        reader = self.reader
        for attr in attrs:
            n = getattr(reader, 'number_of_%s_in_file'%attr)
            method = getattr(reader, 'get_%s_name_in_file'%attr)
            values = [method(x) for x in range(n)]
            if values:
                trait = Trait(values[0], TraitPrefixList(values))
            else:
                trait = Trait('', TraitPrefixList(['']))
                
            self.add_trait('%s_name'%attr, trait)

    
    ######################################################################
    # `TreeNodeObject` interface
    ######################################################################
    def tno_get_label(self, node):
        """ Gets the label to display for a specified object.
        """
        fname = basename(self._file_path.get())
        ret = "CitcomS VTK file (%s)"%fname
        if len(self.file_list) > 1:
            return ret + " (timeseries)"
        else:
            return ret

    ######################################################################
    # Non-public interface
    ######################################################################
    
     ######################################################################
    # `FileDataSource` interface
    ######################################################################
    def initialize(self, base_file_name):
        """Given a single filename which may or may not be part of a
        time series, this initializes the list of files.  This method
        need not be called to initialize the data.
        """
        self.file_list = get_file_list(base_file_name)
        ################
        #Read Meta information
        meta = ""
        try:
            vtk = open(base_file_name, "r")
            vtk.readline()
            meta = vtk.readline()
        except IOError:
            print 'cannot open file'
        try:
            ##
            m = re.search('(?<=NX:)\d+', meta)
            self.nx = int(m.group(0))
            ##
            m = re.search('(?<=NY:)\d+', meta)
            self.ny = int(m.group(0))
            ##
            m = re.search('(?<=NZ:)\d+', meta)
            self.nz = int(m.group(0))
            ##
            m = re.search('(?<=Radius_Inner:)(\d+|.)+', meta)
            self.radius_inner = float(m.group(0))
            ##
        except ValueError:
            print "Non-valid meta information in file..."
    
        vtk.close()
        if len(self.file_list) == 0:
            self.file_list = [base_file_name]
        try:
            self.timestep = self.file_list.index(base_file_name)
        except ValueError:
            self.timestep = 0
        
    
    ######################################################################
    # Non-public interface
    ######################################################################    
    def _file_list_changed(self, value):
        # Change the range of the timestep suitably to reflect new list.
        n_files = len(self.file_list)
        timestep = min(self.timestep, n_files)
        trait = Range(0, n_files - 1, timestep)
        self.add_trait('timestep', trait)
        if self.timestep == timestep:
            self._timestep_changed(timestep)
        else:
            self.timestep = timestep

    def _file_list_items_changed(self, list_event):
        self._file_list_changed(self.file_list)

    def _timestep_changed(self, value):
        file_list = self.file_list
        if len(file_list):
            self._file_path = FilePath(file_list[value])
        else:
            self._file_path = FilePath('')
    
    def __file_path_changed(self, fpath):
        value = fpath.get()
        if not value:
            return
        else:
            self.reader.file_name = value
            self.update_data()
            self.update()
            
            # Setup the outputs by resetting self.outputs.  Changing
            # the outputs automatically fires a pipeline_changed
            # event.
            try:
                n = self.reader.number_of_outputs
            except AttributeError: # for VTK >= 4.5
                n = self.reader.number_of_output_ports
            outputs = []
            for i in range(n):
                outputs.append(self.reader.get_output(i))
            self.outputs = outputs

            # Fire data_changed just in case the outputs are not
            # really changed.  This can happen if the dataset is of
            # the same type as before.
            self.data_changed = True

            # Fire an event so that our label on the tree is updated.
            self.trait_property_changed('name', '',
                                        self.tno_get_label(None))

    def _set_data_name(self, data_type, value):
        if not value or not data_type:
            return
        reader = self.reader
        setattr(reader, data_type, value)
        self.update()
        # Fire an event, so the changes propagate.
        self.data_changed = True

    def _scalars_name_changed(self, value):
        self._set_data_name('scalars_name', value)

    def _vectors_name_changed(self, value):
        self._set_data_name('vectors_name', value)

    def _tensors_name_changed(self, value):
        self._set_data_name('tensors_name', value)

    def _normals_name_changed(self, value):
        self._set_data_name('normals_name', value)

    def _t_coords_name_changed(self, value):
        self._set_data_name('t_coords_name', value)

    def _field_data_name_changed(self, value):
        self._set_data_name('field_data_name', value)
