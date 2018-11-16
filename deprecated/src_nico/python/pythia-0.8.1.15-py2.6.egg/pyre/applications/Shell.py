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


from pyre.inventory.Configurable import Configurable


try:
    # XXX: This default is annoying when one is debugging CGI apps.
    import IPython.ultraTB
    defaultExceptHook = "ultraTB"
except ImportError:
    defaultExceptHook = "current"


class Shell(Configurable):


    import pyre.inventory
    version = pyre.inventory.bool("version")
    
    import pyre.hooks
    excepthook = pyre.hooks.facility("excepthook", family="excepthook",
                                     default=defaultExceptHook)

    import journal
    journal = journal.facility()
    journal.meta['tip'] = 'the logging facility'

    from Preprocessor import Preprocessor
    pp = pyre.inventory.facility("macros", factory=Preprocessor, args=["macros"])


    def __init__(self, app):
        Configurable.__init__(self, app.name)

        self.app = app
        self.registry = None


    def run(self, *args, **kwds):

        app = self.app

        # build storage for the user input
        registry = app.createRegistry()

        # command line
        argv = app.getArgv(*args, **kwds)
        commandLine = app.processCommandline(registry, argv)
        action = commandLine.action
        app.argv = commandLine.processed
        app.unprocessedArguments = commandLine.unprocessed

        # curator
        curator = app.createCurator()
        app.initializeCurator(curator, registry)
        self.setCurator(curator)

        # config context
        context = app.newConfigContext()

        # look for settings
        self.initializeConfiguration(context)
        
        # Temporarily set the app's registry to my own, so that
        # updateConfiguration() will work in readParameterFiles() and
        # collectUserInput().
        app.inventory._priv_registry = self.inventory._priv_registry

        # read parameter files given on the command line
        app.readParameterFiles(registry, context)

        # give descendants an opportunity to collect input from other (unregistered) sources
        app.collectUserInput(registry, context)

        # split the configuration in two
        self.inventory._priv_registry, app.inventory._priv_registry = self.filterConfiguration(self.inventory._priv_registry)
        registry, app.registry = self.filterConfiguration(registry)
        self.registry = registry

        # update user options from the command line
        self.updateConfiguration(registry)

        # transfer user input to my inventory
        self.applyConfiguration(context)

        uc = context.puntUnknownComponents()

        # verify that my input did not contain any typos
        if not context.verifyConfiguration(self, 'strict'):
            import sys
            sys.exit("%s: configuration error(s)" % self.name)

        # initialize the trait cascade
        self.init()

        import sys
        if self.excepthook:
            sys.excepthook = self.excepthook.excepthook

        # ~~ configure the application ~~~
        
        # start fresh
        context = app.newConfigContext()
        context.receiveUnknownComponents(uc) # well, almost fresh

        # update user options from the command line
        app.updateConfiguration(app.registry)
        
        # enable macro expansion
        self.pp.updateMacros(kwds.get('macros', {}))
        context.pp = self.pp

        # transfer user input to the app's inventory
        app.applyConfiguration(context)

        # the main application behavior
        action = action and getattr(app, action)
        if action:
            action()
        elif self.version:
            app.version()
        elif app._helpRequested:
            app.help()
        elif app.verifyConfiguration(context, app.inventory.typos):
            app.init()

            # print a startup page
            app.generateBanner()

            message = kwds.get('message', 'execute')
            method = getattr(app, message)
            method(*args, **kwds)
        else:
            if context.showUsage:
                app.usage()
            import sys
            sys.exit("%s: configuration error(s)" % app.name)

        # shutdown
        app.fini()
        self.fini()

        return


# end of file
