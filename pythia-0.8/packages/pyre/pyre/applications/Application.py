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
from Executive import Executive


class Application(Component, Executive):


    class Inventory(Component.Inventory):

        import pyre.inventory

        typos = pyre.inventory.str(
            name='typos', default='strict',
            validator=pyre.inventory.choice(['relaxed', 'strict', 'pedantic']))
        typos.meta['tip'] = 'specifies the handling of typos in the names of properties and facilities'

        import pyre.weaver
        weaver = pyre.inventory.facility("weaver", factory=pyre.weaver.weaver)
        weaver.meta['tip'] = 'the pretty printer of my configuration as an XML document'

        import journal
        journal = journal.facility()
        journal.meta['tip'] = 'the logging facility'


    def run(self, *args, **kwds):

        # build storage for the user input
        registry = self.createRegistry()
        self.registry = registry

        # command line
        help, self.argv = self.processCommandline(registry)

        # curator
        curator = self.createCurator()
        self.initializeCurator(curator, registry)

        # look for my settings
        self.initializeConfiguration()

        # give descendants an opportunity to collect input from other (unregistered) sources
        self.collectUserInput(registry)

        # update user options from the command line
        self.updateConfiguration(registry)

        # transfer user input to my inventory
        unknownProperties, unknownComponents = self.applyConfiguration()

        # initialize the trait cascade
        self.init()

        # print a startup page
        self.generateBanner()

        # the main application behavior
        if help:
            self.help()
        elif self._showHelpOnly:
            pass
        elif self.verifyConfiguration(unknownProperties, unknownComponents, self.inventory.typos):
            self.execute(*args, **kwds)

        # shutdown
        self.fini()

        return


    def initializeCurator(self, curator, registry):
        if registry is not None:
            curator.config(registry)
            
        # install the curator
        self.setCurator(curator)

        # adjust the depositories
        # first, register the application specific depository
        curator.depositories += self.inventory.getDepositories()
        # then, any extras specified by my descendants
        curator.addDepositories(*self._getPrivateDepositoryLocations())

        return curator


    def collectUserInput(self, registry):
        """collect user input from additional sources"""
        return


    def generateBanner(self):
        """print a startup screen"""
        return


    def __init__(self, name, facility=None):
        if facility is None:
            facility = "application"
            
        Component.__init__(self, name, facility)
        Executive.__init__(self)
    
        # my name as seen by the shell
        import sys
        self.filename = sys.argv[0]

        # commandline arguments left over after parsing
        self.argv = []

        # the user input
        self.registry = None

        # the code generator
        self.weaver = None

        return


    def _init(self):
        Component._init(self)
        self.weaver = self.inventory.weaver

        renderer = self.getCurator().codecs['pml'].renderer
        self.weaver.renderer = renderer

        return


    def _getPrivateDepositoryLocations(self):
        return []



# version
__id__ = "$Id: Application.py,v 1.6 2005/04/05 21:34:12 aivazis Exp $"

# End of file 
