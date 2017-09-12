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


from SimpleComponentHarness import SimpleComponentHarness


class ComponentHarness(SimpleComponentHarness):
    """a mixin class used to create a component harness which is itself a component"""


    def updateConfiguration(self, registry):
        """divide settings between myself and the harnessed component"""
        
        myRegistry, yourRegistry = self.filterConfiguration(registry)
        self.componentRegistry.update(yourRegistry)
        return super(ComponentHarness, self).updateConfiguration(myRegistry)


    def _fini(self):
        """finalize the component"""
        
        if self.component:
            self.component.fini()

        return


    def prepareComponentCurator(self):
        """prepare the persistent store manager for the harnessed component"""

        # the host component has a notion of its persistent store that
        # it wants to share with the harnessed component
        return self.getCurator()
        

    def prepareComponentConfiguration(self, component):
        """prepare the settings for the harnessed component"""

        # the host component has a registry with settings for the
        # harnessed component
        registry = self.componentRegistry
        registry.name = component.name

        return registry


    def createComponentRegistry(self):
        """create a registry instance to store a configuration for the harnessed component"""
        
        return self.createRegistry()


    def __init__(self):
        super(ComponentHarness, self).__init__()
        self.componentRegistry = self.createComponentRegistry()
        return


# version
__id__ = "$Id: ComponentHarness.py,v 1.2 2005/03/11 07:00:17 aivazis Exp $"

# End of file 
