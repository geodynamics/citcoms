#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.components.Component import Component

class CitcomComponent(Component):


    def initialize(self, Module, all_variables):
        self.CitcomModule = Module
        self.all_variables = all_variables
        return



    def setProperties(self):
        return


# version
__id__ = "$Id: CitcomComponent.py,v 1.3 2003/08/27 20:52:47 tan2 Exp $"

# End of file
