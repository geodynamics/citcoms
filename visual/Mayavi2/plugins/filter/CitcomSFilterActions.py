"""Actions to start the CitcomS filter.

"""
# Author: Martin Weier
# Copyright (c) 
# License: 

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
