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


    def __init__(self, name, facility, CitcomModule):
        Component.__init__(self, name, facility)
	self.CitcomModule = CitcomModule
        return



    def setProperties(self, all_variables, PropertySetter):
        self.all_variables = all_variables
	PropertySetter(self.all_variables, self.inventory)
        return



# version
__id__ = "$Id: CitcomComponent.py,v 1.2 2003/08/19 21:24:35 tan2 Exp $"

# End of file
