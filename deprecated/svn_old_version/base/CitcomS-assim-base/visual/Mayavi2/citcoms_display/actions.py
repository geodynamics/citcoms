"""
Actions for MayaVi2 UI
"""

#Author: Martin Weier
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


# Standard library imports
from os.path import isfile

# Enthought library imports
from enthought.pyface import FileDialog, OK

# Mayavi plugin imports
from enthought.mayavi.script import get_imayavi
from enthought.mayavi.core.common import error
from enthought.mayavi.action.common import WorkbenchAction, get_imayavi  # TODO: fix double import of get_imayavi


class OpenVTKAction(WorkbenchAction):
    """
    Open a VTK file.
    """
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
    """
    Open an HDF5 file.
    """
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


class ReduceFilterAction(WorkbenchAction):
    """
    Add a ReduceFilter to the mayavi pipeline.
    """
    def perform(self):
        """ Performs the action. """
        from citcoms_display.plugins.ReduceFilter import ReduceFilter
        f = ReduceFilter()
        mv = get_imayavi(self.window)
        mv.add_filter(f)


class ShowCapsFilterAction(WorkbenchAction):
    """
    Add a ShowCapsFilter to the mayavi pipeline
    """
    def perform(self):
        """ Performs the action. """
        from citcoms_display.plugins.ShowCapsFilter import ShowCapsFilter
        f = ShowCapsFilter()
        mv = get_imayavi(self.window)
        mv.add_filter(f)


