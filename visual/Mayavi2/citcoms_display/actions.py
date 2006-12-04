# Standard library imports
from os.path import isfile

# Enthought library imports
from enthought.pyface import FileDialog, OK

# Mayavi plugin imports
from enthought.mayavi.script import get_imayavi
from enthought.mayavi.core.common import error
from enthought.mayavi.action.common import WorkbenchAction, get_imayavi  # TODO: fix double import of get_imayavi


class OpenVTKAction(WorkbenchAction):
    """ Open a VTK file. """

    def perform(self):
        """Performs the action. """
        wildcard = 'VTK files (*.vtk)|*.vtk|' + FileDialog.WILDCARD_ALL
        parent = self.window.control
        dialog = FileDialog(parent=parent,
                            title='Open CitcomS VTK file',
                            action='open',
                            wildcard=wildcard)
        if dialog.open() == OK:
            if isfile(dialog):
                from citcoms_display.plugins.VTKFileReader import VTKFileReader
                r = VTKFileReader()
                r.initialize(dialog.path)
                mv = get_imayavi(self.window)
                mv.add_source(r)
            else:
                error("File '%s' does not exist!" % dialog.path, parent)
        return


class OpenHDF5Action(WorkbenchAction):
    """ Open an HDF5 file. """

    def perform(self):
        """ Performs the action. """
        wildcard = 'HDF5 files (*.h5)|*.h5|' + FileDialog.WILDCARD_ALL
        parent = self.window.control
        dialog = FileDialog(parent=parent,
                            title='Open CitcomS HDF5 file',
                            action='open',
                            wildcard=wildcard)
        if dialog.open() == OK:
            if isfile(dialog.path):
                from citcoms_display.plugins.HDF5FileReader import HDF5FileReader
                r = HDF5FileReader()
                r.initialize(dialog.path)
                mv = get_imayavi(self.window)
                mv.add_source(r)
            else:
                error("File '%s' does not exist!" % dialog.path, parent)
        return

