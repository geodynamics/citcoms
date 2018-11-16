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


from pyre.inventory.odb.Inventory import Inventory as OdbInventory
from pyre.inventory.odb.Registry import Registry as OdbRegistry


class RegToDictConverter(object):

    """Converts Pyre Registries to ordinary Python dictionaries.

    """
    
    
    def convert(self, document):
        return document.identify(self)
    
    
    def onInventory(self, inventory):
        return self.onRegistry(inventory)
    
    
    def onRegistry(self, registry):
        if not registry.properties and not registry.facilities:
            return {}
        dct = {}
        for name, descriptor in registry.properties.iteritems():
            if name in registry.facilities:
                pass
            else:
                dct[name] = (descriptor.value, locatorRepr(descriptor.locator))
        for facility in registry.facilities:
            component = registry.getFacility(facility)
            if component:
                dct[component.name] = component.identify(self)
        return dct


def dictFromReg(registry):
    converter = RegToDictConverter()
    return converter.convert(registry)


def regFromDict(dct, name="root", cls=OdbRegistry):
    reg = cls(name)
    context = {}
    getLocatorContext(context)
    for k, v in dct.iteritems():
        if isinstance(v, dict):
            reg.attachNode(regFromDict(v, k))
        else:
            value, locator = v
            locator = eval(locator, context)
            reg.setProperty(k, value, locator)
    return reg


def invFromDict(dct, name="inventory"):
    inv = Inventory()
    reg = regFromDict(dct[name])
    inv.updateConfiguration(reg)
    return inv


def registryRepr(registry):
    import pprint
    dct = dictFromReg(registry)
    pprinter = pprint.PrettyPrinter(width=1)
    return pprinter.pformat(dct)


def locatorRepr(locator):
    """Return a __repr__ for a Pyre locator."""
    slots = locator.__slots__
    if not isinstance(slots, tuple):
        slots = (slots, )
    args = [getattr(locator, slot) for slot in slots]
    return locator.__class__.__name__ + str(tuple(args))


def getLocatorContext(dct):
    for className in ['FileLocator', 'ScriptLocator', 'SimpleFileLocator', 'SimpleLocator']:
        m = __import__('pyre.parsing.locators.' + className, globals(), locals(), [className])
        dct[className] = getattr(m, className)
    return


def getNodeWithPath(node, path):
    if isinstance(path, basestring):
        path = path.split('.')
    if len(path) == 0:
        return node
    key = path[0]
    return getNodeWithPath(node.getNode(key), path[1:])


def getPropertyWithPath(root, path, default=''):
    if isinstance(path, basestring):
        path = path.split('.')
    node = getNodeWithPath(root, path[:-1])
    return node.getProperty(path[-1], default)


def getDescriptorWithPath(root, path, default=''):
    if isinstance(path, basestring):
        path = path.split('.')
    node = getNodeWithPath(root, path[:-1])
    return node.properties[path[-1]]


def setPropertyWithPath(root, path, value, locator):
    if isinstance(path, basestring):
        path = path.split('.')
    node = getNodeWithPath(root, path[:-1])
    return node.setProperty(path[-1], value, locator)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


if __name__ == "__main__":
    
    from pyre.applications import Script
    
    class UtilTest(Script):

        componentName = "UtilTest"

        import pyre.inventory as pyre
        answer = pyre.int("answer", default=42)

        def main(self, *args, **kwds):
            print "the answer is", self.answer
            configuration = self.retrieveConfiguration()
            print
            print "the configuration is:"
            print "\n".join([str(item) for item in configuration.render()])
            print
            utilPml = "util.pml"
            print "dumping configuration to", utilPml
            pml = open(utilPml, "w")
            print >> pml, "\n".join(self.weaver.render(configuration))
            pml.close()
            dct = dictFromReg(configuration)
            print
            print "converted configuration to dict:", dct
            print
            print "converted dict back to registry:"
            reg = regFromDict(dct, name=self.name, cls=OdbInventory)
            print "\n".join([str(item) for item in reg.render()])
            print
            utilPml = "util2.pml"
            print "dumping converted registry to", utilPml
            pml = open(utilPml, "w")
            print >> pml, "\n".join(self.weaver.render(reg))
            pml.close()

    
    script = UtilTest()
    script.run()


# end of file
