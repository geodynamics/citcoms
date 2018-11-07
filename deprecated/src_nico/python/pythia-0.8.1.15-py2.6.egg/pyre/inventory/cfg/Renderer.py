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


from pyre.weaver.mills.ConfigMill import ConfigMill


class Renderer(ConfigMill):


    def render(self, inventory):
        document = self.weave(inventory)
        return document


    # handlers

    def onInventory(self, inventory):
        self._rep += ['', '# inventory', '']

        for facility in inventory.facilities.itervalues():
            facility.identify(self)

        return

    
    def onRegistry(self, registry):

        # bail out of empty registries
        if not registry.properties and not registry.facilities:
            return

        self.path.append(registry.name)
        
        self._write('[%s]' % '.'.join(self.path))

        for trait in registry.properties:
            value = registry.getProperty(trait)
            if trait in registry.facilities:
                self._write('%s = %s' % (trait, value))
            else:
                self._write('%s = %s' % (trait, value))
                
        self._write('')
        
        for facility in registry.facilities:
            component = registry.getFacility(facility)
            if component:
                component.identify(self)

        self.path.pop()

        return


    def __init__(self):
        ConfigMill.__init__(self)
        self.path = []
        return


    def _renderDocument(self, document):
        return document.identify(self)


# end of file
