"""Actions to start the CitcomS filter.

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

# Enthought library imports.
# Local imports.
from enthought.mayavi.action.common import WorkbenchAction, get_imayavi

class CitcomSreduce(WorkbenchAction):
    """ An action that starts a delaunay 2d filter. """

    ###########################################################################
    # 'Action' interface.
    ###########################################################################

    def perform(self):
        """ Performs the action. """
        from CitcomSreduce import CitcomSreduce
        f = CitcomSreduce()
        mv = get_imayavi(self.window)
        mv.add_filter(f)

class CitcomSshowCaps(WorkbenchAction):
    """ An action that starts a delaunay 2d filter. """

    ###########################################################################
    # 'Action' interface.
    ###########################################################################

    def perform(self):
        """ Performs the action. """
        from CitcomSshowCaps import CitcomSshowCaps
        f = CitcomSshowCaps()
        mv = get_imayavi(self.window)
        mv.add_filter(f)
