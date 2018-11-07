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


from pyre.parsing.locators.Traceable import Traceable


class Configurable(Traceable):


    # lifecycle management
    def init(self):
        """load user input, initialize my subcomponents and call the custom initialization hook"""

        # initialize my subcomponents
        self.inventory.init()

        # perform any last initializations
        self._init()
        
        return


    def fini(self):
        """call the custom finalization hook and then shut down my subcomponents"""
        
        self._fini()
        self.inventory.fini()
        
        return


    # configuration management
    def retrieveConfiguration(self, registry=None):
        """place my current configuration in the given registry"""

        if registry is None:
            registry = self.createRegistry()

        return self.inventory.retrieveConfiguration(registry)


    def initializeConfiguration(self, context):
        """initialize my private registry using my private settings"""
        return self.inventory.initializeConfiguration(context)


    def loadConfiguration(self, filename):
        """open the given filename and retrieve registry settings for me"""
        return self.inventory.loadConfiguration(filename)


    def updateConfiguration(self, registry):
        """load the user setting in <registry> into my inventory"""
        return self.inventory.updateConfiguration(registry)


    def applyConfiguration(self, context=None):
        """transfer user settings to my inventory"""

        if context is None:
            context = self.newConfigContext()

        context.configureComponent(self)
        
        # give descendants a chance to adjust to configuration changes
        self._configure()
        
        return context


    def filterConfiguration(self, registry):
        """split <registry> in two, according to which traits are in my inventory"""

        myRegistry = self.createRegistry()
        yourRegistry = self.createRegistry()
        yourRegistry.update(registry)

        # Filter-out my properties.
        for trait in self.inventory.properties():
            name = trait.name
            descriptor = registry.properties.get(name)
            if descriptor:
                myRegistry.setProperty(name, descriptor.value, descriptor.locator)
                yourRegistry.deleteProperty(name)

        # Steal nodes which belong to my components.
        for trait in self.inventory.components():
            for name in trait.aliases:
                node = yourRegistry.extractNode(name)
                if node:
                    myRegistry.attachNode(node)

        return myRegistry, yourRegistry


    def newConfigContext(self):
        from ConfigContext import ConfigContext
        return ConfigContext()


    def configureProperties(self, context):
        """set the values of all the properties and facilities in my inventory"""
        self.inventory.configureProperties(context)


    def configureComponents(self, context):
        """guide my subcomponents through the configuration process"""
        self.inventory.configureComponents(context)


    def getDepositories(self):
        return self.inventory.getDepositories()

    # single component management
    def retrieveComponent(self, name, factory, args=(), encodings=['odb'], vault=[], extras=[]):
        """retrieve component <name> from the persistent store"""
        return self.inventory.retrieveComponent(name, factory, args, encodings, vault, extras)


    def configureComponent(self, component, context=None, registry=None):
        """guide <component> through the configuration process"""
        
        if context is None:
            context = self.newConfigContext()
        
        self.inventory.configureComponent(component, context, registry)

        # for backwards compatibility, return the traditional "up, uc" pair
        return context.unknownTraits()


    def collectDefaults(self, registry=None):
        """return a registry containing my default values"""
        if registry is None:
            registry = self.createRegistry()
        return self.inventory.collectDefaults(registry)


    # resource management
    def retrieveObject(self, name, symbol, encodings, vault=[], extras=[]):
        """retrieve object <name> from the persistent store"""
        return self.inventory.retrieveObject(name, symbol, encodings, vault, extras)


    def retrieveTemplate(self, name, vault=[], extras=[]):
        return self.retrieveComponent(name, 'template', vault=vault, extras=extras)


    # vault accessors
    def getVault(self):
        """return the address of my vault"""
        return self.inventory.getVault()


    def setVault(self, vault):
        """set the address of my vault"""
        return self.inventory.setVault(vault)


    # curator accessors
    def getCurator(self):
        """return my persistent store manager"""
        return self.inventory.getCurator()


    def setCurator(self, curator):
        """set my persistent store manager"""
        return self.inventory.setCurator(curator)


    # accessors for the inventory items by category
    def properties(self):
        """return a list of all the property objects in my inventory"""
        return self.inventory.properties()


    def facilities(self):
        """return a list of all the facility objects in my inventory"""
        return self.inventory.facilities()

        
    def components(self):
        """return a list of all the components in my inventory"""
        return self.inventory.components()


    # access to trait values and descriptors by name
    # used by clients that obtain a listing of these names
    # and want to access the underlying objects
    def getTraitValue(self, name):
        try:
            return self.inventory.getTraitValue(name)
        except KeyError:
            pass

        raise AttributeError("object '%s' of type '%s' has no attribute '%s'" % (
            self.name, self.__class__.__name__, name))
        

    def getTraitDescriptor(self, name):
        try:
            return self.inventory.getTraitDescriptor(name)
        except KeyError:
            pass

        raise AttributeError("object '%s' of type '%s' has no attribute '%s'" % (
            self.name, self.__class__.__name__, name))


    # support for the help facility
    def showProperties(self):
        """print a report describing my properties"""
        facilityNames = self.inventory.facilityNames()
        propertyNames = self.inventory.propertyNames()
        propertyNames.sort()
        
        print "properties of %r:" % self.name
        for name in propertyNames:
            if name in facilityNames:
                continue
            
            # get the trait object
            trait = self.inventory.getTrait(name)
            # get the common trait attributes
            traitType = trait.type
            default = trait.default
            meta = trait.meta
            validator = trait.validator
            try:
                tip = meta['tip']
            except KeyError:
                tip = '(no documentation available)'

            # get the trait descriptor from the instance
            descriptor = self.inventory.getTraitDescriptor(name)
            # extract the instance specific values
            value = descriptor.value
            locator = descriptor.locator

            print "    %s=<%s>: %s" % (name, traitType, tip)
            print "        default value: %r" % default
            print "        current value: %r, from %s" % (value, locator)
            if validator:
                print "        validator: %s" % validator

        return


    def showComponents(self):
        facilityNames = self.inventory.facilityNames()
        facilityNames.sort()

        print "facilities of %r:" % self.name
        for name in facilityNames:

            # get the facility object
            facility = self.inventory.getTrait(name)
            meta = facility.meta
            try:
                tip = meta['tip']
            except KeyError:
                tip = '(no documentation available)'

            # get the trait descriptor from the instance
            descriptor = self.inventory.getTraitDescriptor(name)
            # extract the instance specific values
            value = descriptor.value
            locator = descriptor.locator

            print "    %s=<component name>: %s" % (name, tip)
            print "        current value: %r, from %s" % (value.name, locator)
            print "        configurable as: %s" % ", ".join(value.aliases)

        return


    def showUsage(self):
        """print a high level usage screen"""
        propertyNames = self.inventory.propertyNames()
        propertyNames.sort()
        facilityNames = self.inventory.facilityNames()
        facilityNames.sort()

        print "component %r" % self.name

        if propertyNames:
            print "    properties:", ", ".join(propertyNames)

        if facilityNames:
            print "    facilities:", ",".join(facilityNames)

        print "For more information:"
        print "  --help-properties: prints details about user settable properties"
        print "  --help-components: prints details about user settable facilities and components"

        return


    def showCurator(self):
        """print a description of the manager of my persistence store"""
        self.inventory.dumpCurator()
        return


    # default implementations of the various factories
    def createRegistry(self, name=None):
        """create a registry instance to store my configuration"""
        if name is None:
            name = self.name
            
        import pyre.inventory
        return pyre.inventory.registry(name)


    def createInventory(self):
        """create my inventory instance"""
        return self.Inventory(self.name)


    def createMetaInventory(self):
        """create my meta-inventory instance"""
        from MetaInventory import MetaInventory
        return MetaInventory(self.inventory)


    def __init__(self, name=None):
        Traceable.__init__(self)

        if name is None:
            name = self.name # class attribute
        else:
            self.name = name
        self.inventory = self.createInventory()
        
        # provide simple, convenient access to descriptors
        self.metainventory = self.createMetaInventory()

        # other names by which I am known for configuration purposes
        self.aliases = [ name ]

        import journal
        self._debug = journal.debug(name)
        self._info = journal.info(name)
        self._error = journal.error(name)
        self._warning = journal.warning(name)

        # modify the inventory defaults that were hardwired at compile time
        # gives derived components an opportunity to modify their default behavior
        # from what was inherited from their parent's inventory
        self._defaults()
        
        return


    def __getstate__(self):
        # copy the dictionary, since we change it
        odict = self.__dict__.copy()

        # convert inventory to picklable form
        from Inventory import Inventory
        from copy import copy
        inventory = copy(odict['inventory'])
        inventory.__class__ = Inventory
        odict['inventory'] = inventory
        del odict['metainventory']
        
        return odict


    def __setstate__(self, dict):
        self.__dict__.update(dict)
        self.inventory.__class__ = self.Inventory
        self.metainventory = self.createMetaInventory()
        return


    # default implementations for the lifecycle management hooks
    def _defaults(self):
        """modify the default inventory values"""
        return


    def _validate(self, context):
        """perform complex validation involving multiple properties"""
        return


    def _configure(self):
        """modify the configuration programmatically"""
        return


    def _init(self):
        """wake up"""
        return


    def _fini(self):
        """all done"""
        return


    # inventory
    from Inventory import Inventory


    # metaclass
    from ConfigurableClass import ConfigurableClass
    __metaclass__ = ConfigurableClass


# version
__id__ = "$Id: Configurable.py,v 1.5 2005/03/27 01:22:41 aivazis Exp $"

# End of file 
