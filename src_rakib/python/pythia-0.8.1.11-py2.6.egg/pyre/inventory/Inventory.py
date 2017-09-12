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


class Inventory(object):


    def initializeConfiguration(self, context):
        # load my settings from the persistent store
        # NYI: load options based on my facility as well?
        #      it might be useful when we auto-generate settings for an entire group of clients
        #      whose names may not be known when their configurations are built
        self._priv_registry = self._priv_curator.getTraits(
            self._priv_name, context,
            vault=self._priv_vault,
            extraDepositories=self._priv_depositories)

        return


    def loadConfiguration(self, filestem):
        """load the registry contained in the given pml file (without the extension)"""

        import pyre.inventory
        codec = pyre.inventory.codecPML()
        shelf = codec.open(filestem)

        return shelf['inventory']


    def updateConfiguration(self, registry):
        return self._priv_registry.update(registry)


    def configureProperties(self, context):
        """configure my properties using user settings in my registry"""

        # Merge defaults, so that they will be subject to macro
        # expansion.  This also forces the initialization of all
        # properties, which is significant if there are bogus defaults
        # (a prime example being a nonexistant pathname for an
        # InputFile).
        registry = self.collectPropertyDefaults()
        registry.update(self._priv_registry)

        # loop over the registry property entries and
        # attempt to set the value of the corresponding inventory item
        for name, descriptor in registry.properties.items():
            prop = self._traitRegistry.get(name, None)
            if prop:
                try:
                    context.setProperty(prop, self, descriptor.value, descriptor.locator)
                except SystemExit:
                    raise
                except Exception, error:
                    from Item import Item
                    context.error(error, items=[Item(prop, descriptor)])
            else:
                context.unrecognizedProperty(name, descriptor.value, descriptor.locator)
                self._priv_registry.deleteProperty(name)

        return


    def configureComponents(self, context):
        """configure my components using options from my registry"""

        myComponents = self.components(context)

        aliases = {}
        for component in myComponents:
            # associate a persistent store with every subcomponent
            component.setCurator(self._priv_curator)
            component.initializeConfiguration(context)

            # construct a list of the public names of this component
            # setting are overriden from left to right
            componentAliases = list(component.aliases)
            componentAliases.reverse()
            # for each registered public name of this component
            for alias in componentAliases:
                # register this name so we can hunt down typos
                aliases[alias] = component

                registry = self._priv_registry.getFacility(alias)
                if registry:
                    component.updateConfiguration(registry)

            component.applyConfiguration(context)

        # loop over the registry facility entries and
        # update the configuration of all the named components/facilities
        # note that this only affects components for which there are settings in the registry
        # this is done in a separate loop because it provides an easy way to catch typos
        # on the command line
        for name in self._priv_registry.facilities.keys():
            if not aliases.has_key(name):
                node = self._priv_registry.extractNode(name)
                context.unknownComponent(name, node)

        return


    def retrieveConfiguration(self, registry):
        """place the current inventory configuration in the given registry"""

        from Facility import Facility
        from Property import Property

        node = registry.getNode(self._priv_name)

        for prop in self._traitRegistry.itervalues():

            name = prop.name
            descriptor = self.getTraitDescriptor(name)
            value = descriptor.value
            locator = descriptor.locator

            if value and isinstance(prop, Facility):
                value = value.name

            node.setProperty(name, value, locator)

        for component in self.components():
            component.retrieveConfiguration(node)
            
        return registry


    def collectDefaults(self, registry):
        """place my default values in the given registry"""

        from Facility import Facility

        node = registry.getNode(self._priv_name)

        for prop in self._traitRegistry.itervalues():
            name = prop.name
            value, locator = prop._getDefaultValue(self)
            if isinstance(prop, Facility):
                # This isn't necessarily true.
                value = value.name
            node.setProperty(name, value, locator)

        for facility in self._facilityRegistry.itervalues():
            components = facility._retrieveAllComponents(self)
            for component in components:
                component.setCurator(self._priv_curator)
                component.collectDefaults(node)

        return registry


    def collectPropertyDefaults(self, registry=None):
        """place my default values in the given registry"""

        from Facility import Facility
        from pyre.inventory.odb.Registry import Registry
        import pyre.parsing.locators

        if registry is None:
            registry = Registry(self._priv_name)

        locator = pyre.parsing.locators.default()

        for prop in self._traitRegistry.itervalues():

            # We intentionally don't call _getDefaultValue() -- at
            # this stage, we don't want anything to happen (files to
            # be opened, components to be instantiated, ...)
            try:
                # Pick up any values set by _defaults() methods.
                value = self._getTraitValue(prop.name)
            except KeyError:
                value = prop.default
            
            # The 'isinstance' is a limitation of the framework: e.g.,
            # files and dimensionals to not stringify cleanly.
            # Fortunately, we are only interested in string defaults
            # at present (for macro expansion).
            if isinstance(value, basestring):
                registry.setProperty(prop.name, value, locator)

        return registry


    def configureComponent(self, component, context, registry=None):
        """configure <component> using options from the given registry"""

        # if none were given, let the registry be our own
        if registry is None:
            registry = self._priv_registry

        # set the component's curator
        component.setCurator(self._priv_curator)
        component.initializeConfiguration(context)

        # find any relevant traits in my registry
        # look for facility traits
        aliases = list(component.aliases)
        aliases.reverse()
        for alias in aliases:
            traits = registry.getFacility(alias)
            component.updateConfiguration(traits)

        # apply the settings
        component.applyConfiguration(context)

        return


    def retrieveComponent(
        self, name, factory, args=(), encodings=['odb'], vault=[], extraDepositories=[]):
        """retrieve component <name> from the persistent store"""

        if extraDepositories:
            import journal
            journal.firewall("inventory").log("non-null extraDepositories")

        return self._priv_curator.retrieveComponent(
            name=name, facility=factory, args=args, encodings=encodings,
            vault=vault, extraDepositories=self._priv_depositories)
        

    def retrieveAllComponents(
        self, factory, args=(), encoding='odb', vault=[], extraDepositories=[]):
        """retrieve all possible components for <factory> from the persistent store"""

        if extraDepositories:
            import journal
            journal.firewall("inventory").log("non-null extraDepositories")

        return self._priv_curator.retrieveAllComponents(
            facility=factory, args=args, encoding=encoding,
            vault=vault, extraDepositories=self._priv_depositories)
        

    def retrieveBuiltInComponent(self, name, factory, args=(), vault=[]):
        import pkg_resources
        group = "pyre.odb." + (".".join([self._priv_name] + vault))
        for ep in pkg_resources.iter_entry_points(group, name):
            factory = ep.load()
            component = factory(*args)
            return component
        return None


    def retrieveObject(
        self, name, symbol, encodings, vault=[], extraDepositories=[]):
        """retrieve object <name> from the persistent store"""

        if extraDepositories:
            import journal
            journal.firewall("inventory").log("non-null extraDepositories")

        return self._priv_curator.retrieveObject(
            name=name, symbol=symbol, encodings=encodings,
            vault=vault, extraDepositories=self._priv_depositories)
        

    def init(self):
        """initialize subcomponents"""

        for component in self.components():
            component.init()

        return


    def fini(self):
        """finalize subcomponents"""

        for component in self.components():
            component.fini()

        return


    def showHelp(self):
        for component in self.components():
            component.showHelp()
        return


    # lower level interface
    def getVault(self):
        """return the address of my vault"""
        return self._priv_vault


    def setVault(self, vault):
        """set the address of my vault"""
        assert self._priv_depositories is None # must be called before setCurator()
        self._priv_vault = vault
        return


    def getCurator(self):
        """return the curator that resolves my trait requests"""
        return self._priv_curator


    def setCurator(self, curator):
        """set my persistent store manager and initialize my registry"""

        # keep track of the curator
        self._priv_curator = curator

        # construct my private depositories
        self._priv_depositories = self._createDepositories()

        return


    def dumpCurator(self):
        """print a description of the manager of my persistence store"""
        return self._priv_curator.dump(self._priv_depositories)


    def getDepositories(self):
        """return my private depositories"""
        return self._priv_depositories


    def retrieveShelves(self, address, extension):
        return self._priv_curator.retrieveShelves(
            address, extension, extraDepositories=self._priv_depositories)


    def getTraitDescriptor(self, traitName):
        try:
            return self._getTraitDescriptor(traitName)

        except KeyError:
            pass
        
        self._forceInitialization(traitName)
        return self._getTraitDescriptor(traitName)


    def getTraitValue(self, traitName):
        try:
            return self._getTraitValue(traitName)

        except KeyError:
            pass
        
        return self._forceInitialization(traitName)


    def getTraitLocator(self, traitName):
        try:
            return self._getTraitLocator(traitName)

        except KeyError:
            pass
        
        self._forceInitialization(traitName)
        return self._getTraitLocator(traitName)


    def getTrait(self, traitName):
        return self._traitRegistry[traitName]


    # accessors for the inventory items by category
    def properties(self):
        """return a list of my property objects"""
        return self._traitRegistry.values()


    def propertyNames(self):
        """return a list of the names of all my traits"""
        return self._traitRegistry.keys()


    def facilities(self):
        """return a list of my facility objects"""
        return self._facilityRegistry.values()

        
    def facilityNames(self):
        """return a list of the names of all my facilities"""
        return self._facilityRegistry.keys()


    def components(self, context=None):
        """return a list of my components"""

        from pyre.inventory import Error

        candidates = []

        for name, facility in self._facilityRegistry.iteritems():
            try:
                component = facility.__get__(self)
                if component and component is not Error:
                    candidates.append(component)
            except SystemExit:
                raise
            except Exception, error:
                if context:
                    import sys, traceback, pyre.parsing.locators
                    stackTrace = traceback.extract_tb(sys.exc_info()[2])
                    locator = pyre.parsing.locators.stackTrace(stackTrace)
                    context.error(error, locator=locator)
                else:
                    raise
        
        return candidates


    def __init__(self, name):
        # the name of the configurable that manages me
        self._priv_name = name

        # the name of my vault
        self._priv_vault = []
        
        # the manager of my persistent trait store
        self._priv_curator = None

        # the private depositories
        self._priv_depositories = None

        # the accumulator of user supplied state
        self._priv_registry = None

        # local storage for the descriptors created by the various traits
        self._priv_inventory = {}

        return


    def _createDepositories(self):
        depositories = self._priv_curator.createPrivateDepositories(self._priv_name)
        return depositories
    

    def _getTraitValue(self, name):
        return self._getTraitDescriptor(name).value


    def _setTraitValue(self, name, value, locator):
        descriptor = self._getTraitDescriptor(name)
        descriptor.value = value
        descriptor.locator = locator
        return


    def _initializeTraitValue(self, name, value, locator):
        descriptor = self._createTraitDescriptor()
        descriptor.value = value
        descriptor.locator = locator
        self._setTraitDescriptor(name, descriptor)
        return


    def _getTraitLocator(self, name):
        return self._getTraitDescriptor(name).locator


    def _createTraitDescriptor(self):
        from Descriptor import Descriptor
        return Descriptor()


    def _getTraitDescriptor(self, name):
        return self._priv_inventory[name]


    def _setTraitDescriptor(self, name, descriptor):
        self._priv_inventory[name] = descriptor
        return


    def _forceInitialization(self, name):
        trait = self._traitRegistry[name]
        return trait.__get__(self)


    # trait registries
    _traitRegistry = {}
    _facilityRegistry = {}
    _myTraitRegistry = {}


    # metaclass
    from Notary import Notary
    __metaclass__ = Notary


# version
__id__ = "$Id: Inventory.py,v 1.3 2005/03/11 06:59:08 aivazis Exp $"

# End of file 
