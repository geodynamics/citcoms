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


from ComponentHarness import ComponentHarness


class DynamicComponentHarness(ComponentHarness):


    def createComponent(self):
        """retrieve my harnessed component from the persistent store"""
        
        registry = self.registry
        facilityName = self.getFacilityName(registry)
        componentName = self.getComponentName(facilityName, registry)

        component = self.retrieveComponent(componentName, facilityName)

        if not component:
            raise ValueError("could not locate '%s'(%s)" % (componentName, facilityName))

        return component


    def getFacilityName(self, registry):
        """return the facility implemented by my harnessed component"""

        name = registry.getProperty('facility')
        registry.deleteProperty('facility')

        return name


    def getComponentName(self, facilityName, registry):
        """extract the name of the harnessed component"""

        name = registry.getProperty(facilityName)
        registry.deleteProperty(facilityName)

        return name


# version
__id__ = "$Id: DynamicComponentHarness.py,v 1.2 2005/03/14 22:53:50 aivazis Exp $"

# End of file 
