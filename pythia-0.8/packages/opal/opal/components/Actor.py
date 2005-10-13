#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from pyre.components.Component import Component


class Actor(Component):


    def perform(self, app, routine=None):
        raise NotImplementedError("class %r must override 'perform'" % self.__class__.__name__)


    def __init__(self, name):
        Component.__init__(self, name, facility='actor')
        self.routine = None
        return


# version
__id__ = "$Id: Actor.py,v 1.2 2005/05/02 18:08:26 pyre Exp $"

# End of file 
