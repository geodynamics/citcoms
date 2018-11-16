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


class ConfigContext(object):

    #
    # application interface
    #
    
    def error(self, error, **attributes):
        for k, v in attributes.iteritems():
            setattr(error, k, v)
        if not hasattr(error, 'locator'):
            if hasattr(error, 'items') and len(error.items) == 1:
                error.locator = error.items[0].locator
            else:
                from pyre.parsing.locators import simple
                error.locator = simple("validator")
        if hasattr(error, 'items'):
            for item in error.items:
                # This could be wrong in _validate().  For now, the
                # framework assumes that _validate() is simple and
                # polite: i.e., a component only reports errors for
                # its own traits.
                item.path = self.path + [item.name]
        if not hasattr(error, 'channel'):
            error.channel = 'e'
        self.errors.append(error)
        return


    #
    # private
    #

    def unknownComponent(self, name, registry):
        self.unknownComponents.attachNode(registry)
        self.showUsage = True


    def unrecognizedProperty(self, name, value, locator):
        self.unrecognizedProperties.setProperty(name, value, locator)
        self.showUsage = True


    def configureComponent(self, component):

        # push
        parent = (self.unrecognizedProperties, self.unknownComponents)
        self.unrecognizedProperties = self.unrecognizedProperties.getNode(component.name)
        self.unknownComponents = self.unknownComponents.getNode(component.name)
        self.path.append(component.name)
        
        # apply user settings to the component's properties
        component.configureProperties(self)

        # apply user settings to the component's subcomponents
        component.configureComponents(self)

        # give the component an opportunity to perform complex validation
        component._validate(self)

        # pop
        self.path.pop()
        self.unrecognizedProperties, self.unknownComponents = parent

        return


    def setProperty(self, prop, instance, value, locator):
        if self.pp:
            value = self.pp.expandMacros(value)
        prop._set(instance, value, locator)
        return


    def puntUnknownComponents(self):
        from pyre.inventory import registry
        uc = self.unknownComponents
        self.unknownComponents = registry("inventory")
        return uc


    def receiveUnknownComponents(self, uc):
        self.unknownComponents = uc
        return


    def unknownTraits(self):
        unrecognizedProperties = []
        unknownComponents = []
        
        node = self.unrecognizedProperties
        for path, value, locator in node.allProperties():
            path = '.'.join(path[1:])
            unrecognizedProperties.append((path, value, locator))

        node = self.unknownComponents
        for path, value, locator in node.allProperties():
            path = '.'.join(path[1:-1])
            unknownComponents.append(path)

        return (unrecognizedProperties, unknownComponents)


    def verifyConfiguration(self, component, modeName):
        """verify that the user input did not contain any typos"""

        # Convert all unrecognized properties and unknown components
        # into errors.

        node = self.unrecognizedProperties.getNode(component.name)
        for path, value, locator in node.allProperties():
            self.error(UnrecognizedPropertyError(path, value, locator))

        node = self.unknownComponents
        for path, value, locator in node.allProperties():
            self.error(UnknownComponentError(path, value, locator))

        # Log all configuration errors and warnings.  Determine the
        # severity of property/component typos as a function of the
        # given mode.

        class Channel(object):
            def __init__(self, factory):
                self.channel = factory("pyre.inventory")
                self.tally = 0
            def line(self, message):
                self.channel.line(message)
            def log(self, message=None, locator=None):
                self.channel.log(message, locator)
                self.tally += 1

        import journal
        info     = Channel(journal.info)
        warning  = Channel(journal.warning)
        error    = Channel(journal.error)

        mode = dict(
            relaxed   = dict(up=warning, uc=info,    e=warning),
            strict    = dict(up=error,   uc=warning, e=error),
            pedantic  = dict(up=error,   uc=error,   e=error),
            )
        
        channel = mode[modeName]

        for e in self.errors:
            self.log(channel[e.channel], e)

        return error.tally == 0


    def log(self, channel, error):

        locator = None

        # Perhaps there should be a decorator which normalizes the
        # 'error' interface...

        if hasattr(error, 'locator'):
            locator = error.locator

        if hasattr(error, 'items'):
            for item in error.items:
                path = '.'.join(item.path[1:])
                channel.line("%s <- '%s'" % (path, item.value))
        elif hasattr(error, 'path'):
            path = '.'.join(error.path[1:])
            channel.line("%s <- '%s'" % (path, error.value))
        
        channel.log(error, locator)
        channel.tally = channel.tally + 1
        
        return


    def __init__(self):
        from pyre.inventory import registry
        
        self.unrecognizedProperties = registry("inventory")
        self.unknownComponents = registry("inventory")
        self.path = []

        self.errors = []

        self.pp = None

        self.showUsage = False

        return


class ConfigurationError(Exception):

    def __init__(self, path, value, locator):
        Exception.__init__(self)
        self.path = path
        self.value = value
        self.locator = locator


class UnrecognizedPropertyError(ConfigurationError):

    channel = 'up'
        
    def __str__(self):
        prop = '.'.join(self.path[1:])
        return "unrecognized property '%s'" % prop


class UnknownComponentError(ConfigurationError):
        
    channel = 'uc'
        
    def __str__(self):
        component = '.'.join(self.path[1:-1])
        return "unknown component '%s'" % component
        

# end of file 
