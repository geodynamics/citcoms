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



    def setProperties(self, PropertySetter):
	PropertySetter(self.inventory)
        return



# version
__id__ = "$Id: CitcomComponent.py,v 1.1 2003/07/24 17:46:46 tan2 Exp $"

# End of file
