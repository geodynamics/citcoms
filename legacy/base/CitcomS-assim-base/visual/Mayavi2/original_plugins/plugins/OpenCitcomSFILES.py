# Standard library imports.
from os.path import isfile

# Enthought library imports.
from enthought.pyface import FileDialog, OK

# Local imports
from enthought.mayavi.script import get_imayavi
from enthought.mayavi.core.common import error
from enthought.mayavi.action.common import WorkbenchAction, get_imayavi

######################################################################
# `OpenCitcomSVtkFile` class.
######################################################################
class OpenCitcomSVTKFILE(WorkbenchAction):
    """ An action that opens a new VTK file. """

    ###########################################################################
    # 'Action' interface.
    ###########################################################################

    def perform(self):
        """ Performs the action. """
        wildcard = 'VTK files (*.vtk)|*.vtk|' + FileDialog.WILDCARD_ALL
        parent = self.window.control
        dialog = FileDialog(parent=parent,
                            title='Open CitcomS VTK file',
                            action='open', wildcard=wildcard
                            )
        if dialog.open() == OK:
            if not isfile(dialog.path):
                error("File '%s' does not exist!"%dialog.path, parent)
                return
            from enthought.mayavi.plugins.CitcomS_vtk_file_reader import CitcomSVTKFileReader
            r = CitcomSVTKFileReader()
            r.initialize(dialog.path)
            mv = get_imayavi(self.window)
            mv.add_source(r)

######################################################################
# `OpenCitcomSVtkFile` class.
######################################################################
class OpenCitcomSHDFFILE(WorkbenchAction):
    """ An action that opens a new VTK file. """

    ###########################################################################
    # 'Action' interface.
    ###########################################################################

    def perform(self):
        """ Performs the action. """
        wildcard = 'HDF files (*.h5)|*.h5|' + FileDialog.WILDCARD_ALL
        parent = self.window.control
        dialog = FileDialog(parent=parent,
                            title='Open CitcomS H5 file',
                            action='open', wildcard=wildcard
                            )
        if dialog.open() == OK:
            if not isfile(dialog.path):
                error("File '%s' does not exist!"%dialog.path, parent)
                return
            from enthought.mayavi.plugins.CitcomS_hdf_file_reader import CitcomSHDFFileReader
            r = CitcomSHDFFileReader()
            r.initialize(dialog.path)
            mv = get_imayavi(self.window)
            mv.add_source(r)
