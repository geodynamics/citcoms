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


from pyre.inventory.Configurable import Configurable


class Component(Configurable):


    class Inventory(Configurable.Inventory):

        import pyre.inventory

        usage = pyre.inventory.bool("help", default=False)
        usage.meta['tip'] = 'prints a screen that describes my traits'

        showProperties = pyre.inventory.bool("help-properties", default=False)
        showProperties.meta['tip'] = 'prints a screen that describes my properties'

        showComponents = pyre.inventory.bool("help-components", default=False)
        showComponents.meta['tip'] = 'prints a screen that describes my subcomponents'

        showCurator = pyre.inventory.bool("help-persistence", default=False)
        showCurator.meta['tip'] = 'prints a screen that describes my persistent store'


    def updateConfiguration(self, registry):
        # verify that we were handed the correct registry node
        if registry:
            name = registry.name
            if name not in self.aliases:
                import journal
                journal.firewall("inventory").log(
                    "bad registry node: %s != %s" % (name, self.name))

        return Configurable.updateConfiguration(self, registry)


    def __init__(self, name=None, facility=None):
        Configurable.__init__(self, name)
        #self.facility = facility # not used

        self._helpRequested = False

        return


    def _configure(self):
        Configurable._configure(self)
        if (self.inventory.usage or
            self.inventory.showProperties or
            self.inventory.showComponents or
            self.inventory.showCurator):
            self._helpRequested = True
        else:
            for component in self.components():
                if component._helpRequested:
                    self._helpRequested = True
                    break
        return


    def showHelp(self):
        self.inventory.showHelp()
        self._showHelp()
        return


    def _showHelp(self):
        if self.inventory.usage:
            self.showUsage()

        if self.inventory.showProperties:
            self.showProperties()

        if self.inventory.showComponents:
            self.showComponents()

        if self.inventory.showCurator:
            self.showCurator()

        return


# version
__id__ = "$Id: Component.py,v 1.3 2005/04/05 21:34:48 aivazis Exp $"

# End of file 
