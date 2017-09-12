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


from Trait import Trait
from pyre.inventory import Uninit


class Facility(Trait):


    def __init__(self, name, family=None, default=Uninit, factory=None, args=(), meta=None,
                 vault=None):
        Trait.__init__(self, name, 'facility', default, meta)

        self.args = args
        self.factory = factory

        if family is None:
            family = name
        self.family = family

        if vault is not None:
            self.vault = vault

        return


    def _getDefaultValue(self, instance):
        
        # Initialize my value (preventing further lookups), in case we
        # don't make it out of here alive.
        import pyre.parsing.locators
        from pyre.inventory import Error
        locator = pyre.parsing.locators.error()
        instance._initializeTraitValue(self.name, Error, locator)

        import pyre.parsing.locators
        locator = pyre.parsing.locators.default()

        if not self.default in [None, Uninit]:
            component = self.default
            # if we got a string, resolve
            if isinstance(component, basestring):
                component, loc = self._retrieveComponent(instance, component)
                locator = pyre.parsing.locators.chain(loc, locator)
                
            return component, locator

        if self.factory is not None:
            # instantiate the component
            component = self.factory(*self.args)
            # adjust the configuration aliases to include my name
            aliases = component.aliases
            if self.name not in aliases:
                aliases.append(self.name)
            
            # return
            return component, locator

        component, locator = self._getBuiltInDefaultValue(instance)
        if component is not None:
            return component, locator

        if self.default is Uninit:
            # oops: expect exceptions galore!
            import journal
            firewall = journal.firewall('pyre.inventory')
            firewall.log(
                "facility %r was given neither a default value nor a factory method" % self.name)

        # None is a special value; it means that a facility is not set
        return None, None


    def _getBuiltInDefaultValue(self, instance):
        return None, None


    def _set(self, instance, component, locator):
        if isinstance(component, basestring):
            component, source = self._retrieveComponent(instance, component)

            import pyre.parsing.locators
            locator = pyre.parsing.locators.chain(source, locator)

        if component is None:
            return

        # get the old component
        try:
            old = instance._getTraitValue(self.name)
        except KeyError:
            # the binding was uninitialized
            return instance._initializeTraitValue(self.name, component, locator)

        # if the previous binding was non-null, finalize it
        if old:
            old.fini()
        
        # bind the new value
        return instance._setTraitValue(self.name, component, locator)


    def _retrieveComponent(self, instance, componentName):
        component = instance.retrieveComponent(
            name=componentName,
            factory=self.family,
            vault=self.vault)

        if component is not None:
            locator = component.getLocator()
        else:
            import pyre.parsing.locators
            component = self._retrieveBuiltInComponent(instance, componentName)
            if component is not None:
                locator = pyre.parsing.locators.builtIn()
            else:
                component = self._import(instance, componentName)
                if component:
                    locator = pyre.parsing.locators.simple('imported')
                else:
                    locator = pyre.parsing.locators.simple('not found')
                    return None, locator

        # adjust the names by which this component is known
        component.aliases.append(self.name)
            
        return component, locator


    def _retrieveAllComponents(self, instance):
        return instance.retrieveAllComponents(factory=self.family)


    class Error(Exception):
        def __init__(self, **kwds):
            self.__dict__.update(kwds)


    class ComponentNotFound(Error):
        def __str__(self):
            return "could not bind facility '%(facility)s': component '%(component)s' not found" % self.__dict__


    class FactoryNotFound(Error):
        def __str__(self):
            return "could not bind facility '%(facility)s': no factory named '%(factory)s' in '%(module)s'" % self.__dict__


    class FactoryNotCallable(Error):
        def __str__(self):
            return "could not bind facility '%(facility)s': factory '%(module)s:%(factory)s' is not callable" % self.__dict__


    def _retrieveBuiltInComponent(self, instance, name):
        return instance.retrieveBuiltInComponent(
            name=name,
            factory=self.family,
            vault=self.vault)


    def _import(self, instance, name):

        factoryName = self.family
        path = name.split(':')
        c = len(path)
        if c == 1:
            factoryPath = [factoryName]
        elif c == 2:
            factoryPath = path[1].split('.')
            if not factoryPath[-1]:
                factoryPath.pop()
                if not factoryPath:
                    factoryPath.append(factoryName)
            else:
                factoryPath.append(factoryName)
        else:
            raise Facility.ComponentNotFound(
                facility=self.name, component=name)
        module = path[0]
        factoryName = '.'.join(factoryPath)
        objName = module + ':' + factoryName

        try:
            from pyre.util import loadObject
            factory = loadObject(objName)
        except (ImportError, ValueError):
            raise Facility.ComponentNotFound(
                facility=self.name, component=name)
        except AttributeError:
            raise Facility.FactoryNotFound(
                facility=self.name, module=module, factory=factoryName)

        if not callable(factory):
            raise Facility.FactoryNotCallable(
                facility=self.name, module=module, factory=factoryName)

        item = factory(*self.args)

        return item


    vault = []


    # interface registry
    _interfaceRegistry = {}

    # metaclass
    from Interface import Interface
    __metaclass__ = Interface


# version
__id__ = "$Id: Facility.py,v 1.4 2005/03/29 12:11:33 aivazis Exp $"

# End of file 
