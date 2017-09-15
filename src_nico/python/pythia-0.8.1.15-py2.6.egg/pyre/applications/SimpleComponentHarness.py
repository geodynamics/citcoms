#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                      California Institute of Technology
#                        (C) 2006  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


class SimpleComponentHarness(object):


    def harnessComponent(self):
        """harness an external component"""

        # create the component
        component = self.createComponent()

        # initialize the persistent store used by the component to configure itself
        curator = self.prepareComponentCurator()

        # prepare optional configuration for the component
        registry = self.prepareComponentConfiguration(component)

        # configure the component
        # collect unknown traits for the components and its subcomponents
        context = self.configureHarnessedComponent(component, curator, registry)

        if not context.verifyConfiguration(component, 'strict'):
            return

        # initialize the component
        component.init()

        # register it
        self.component = component

        return component


    def createComponent(self):
        """create the harnessed component"""
        raise NotImplementedError(
            "class %r must override 'createComponent'" % self.__class__.__name__)


    def configureHarnessedComponent(self, component, curator, registry):
        """configure the harnessed component"""

        context = component.newConfigContext()
        
        # link the component with the curator
        component.setCurator(curator)
        component.initializeConfiguration(context)

        # update the component's inventory with the optional settings we
        # have gathered on its behalf
        component.updateConfiguration(registry)

        # load the configuration onto the inventory
        component.applyConfiguration(context)

        return context


    def __init__(self):
        super(SimpleComponentHarness, self).__init__()
        self.component = None
        return


# end of file 
