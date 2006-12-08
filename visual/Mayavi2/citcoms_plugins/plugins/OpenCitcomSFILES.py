# Standard library imports.
from os.path import isfile

# Enthought library imports.
from enthought.pyface import FileDialog, OK

# Local imports
from enthought.mayavi.script import get_imayavi
from enthought.mayavi.core.common import error
from enthought.mayavi.action.common import WorkbenchAction, get_imayavi

######################################################################
# `OpenCitcomSHDFFile` class.
######################################################################
class OpenCitcomSHDFFILE(WorkbenchAction):
    """ An action that opens a new HDF5 file. """

    ###########################################################################
    # 'Action' interface.
    ###########################################################################

    def perform(self):
        """ Performs the action. """
        wildcard = 'HDF files (*.h5)|*.h5|' + FileDialog.WILDCARD_ALL
        parent = self.window.control
        dialog = FileDialog(parent=parent,
                            title='Open CitcomS HDF5 file',
                            action='open', wildcard=wildcard
                            )
        if dialog.open() == OK:
            if not isfile(dialog.path):
                error("File '%s' does not exist!"%dialog.path, parent)
                return
            from citcoms_plugins.plugins.CitcomS_hdf_file_reader import CitcomSHDFFileReader
            r = CitcomSHDFFileReader()
            r.initialize(dialog.path)
            mv = get_imayavi(self.window)
            mv.add_source(r)

