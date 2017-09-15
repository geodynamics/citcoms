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


    name = "application"


    class Inventory(Component.Inventory):

        import pyre.inventory

        typos = pyre.inventory.str(
            name='typos', default='strict',
            validator=pyre.inventory.choice(['relaxed', 'strict', 'pedantic']))
        typos.meta['tip'] = 'specifies the handling of typos in the names of properties and facilities'

        import pyre.weaver
        weaver = pyre.inventory.facility("weaver", factory=pyre.weaver.weaver)
        weaver.meta['tip'] = 'the pretty printer of my configuration as an XML document'


    def run(self, *args, **kwds):
        from Shell import Shell
        shell = Shell(self)
        shell.run(*args, **kwds)
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


    def readParameterFiles(self, registry, context):
        """read parameter files given on the command line"""
        from os.path import isfile, splitext
        import pyre.parsing.locators
        locator = pyre.parsing.locators.commandLine()
        argv = self.argv
        self.argv = []
        for arg in argv:
            base, ext = splitext(arg)
            encoding = ext[1:] # NYI: not quite
            codec = self.getCurator().codecs.get(encoding)
            if codec:
                try:
                    shelf = codec.open(base)
                except Exception, error:
                    context.error(error, locator=locator)
                else:
                    for facilityName, node in shelf['inventory'].facilities.iteritems():
                        if facilityName == self.name:
                            self.updateConfiguration(node)
                        else:
                            context.unknownComponent(facilityName, node)
            else:
                self.argv.append(arg)
        return

    
    def collectUserInput(self, registry, context):
        """collect user input from additional sources"""
        return


    def generateBanner(self):
        """print a startup screen"""
        return


    def entryName(self):
        return self.__class__.__module__ + ':' + self.__class__.__name__


    def path(self):
        """Return the minimal Python search path for this application."""
        import sys
        return sys.path


    def pathString(self):
        return ':'.join(self.path())


    def __init__(self, name=None, facility=None):
        Component.__init__(self, name, facility)
        Executive.__init__(self)
    
        # my name as seen by the shell
        import sys
        self.filename = sys.argv[0]

        # commandline arguments left over after parsing
        self.argv = []
        self.unprocessedArguments = []

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


    def dumpDefaults(self):
        configuration = self.collectDefaults()
        # save the configuation as a PML file
        configPml = self.name + "-defaults.pml"
        pml = open(configPml, 'w')
        print >> pml, "\n".join(self.weaver.render(configuration))
        pml.close()



# version
__id__ = "$Id: Application.py,v 1.6 2005/04/05 21:34:12 aivazis Exp $"

# End of file 
