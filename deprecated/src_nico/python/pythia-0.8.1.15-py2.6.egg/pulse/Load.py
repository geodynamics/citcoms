#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.components.Component import Component


class Load(Component):


    def updatePressure(self, boundary):
        raise NotImplementedError(
            "class '%s' must override 'updatePressure'" % self.__class__.__name__)


    def advance(self, dt):
        raise NotImplementedError(
            "class '%s' must override 'advance'" % self.__class__.__name__)


    def __init__(self, name):
        Component.__init__(self, name, "generator")
        return


# version
__id__ = "$Id: Load.py,v 1.1.1.1 2005/03/08 16:13:57 aivazis Exp $"

# End of file 
