#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                      California Institute of Technology
#                        (C) 2007  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from Item import Item


class MetaInventory(object):


    def __init__(self, inventory):
        self._inventory = inventory


    def __getattr__(self, name):
        trait = getattr(self._inventory.__class__, name)
        descriptor = self._inventory.getTraitDescriptor(trait.name)
        return Item(trait, descriptor)


    def iteritems(self, cls=None):
        if cls is None:
            registry = self._inventory._traitRegistry
        else:
            if not isinstance(self._inventory, cls):
                raise TypeError("inventory object is not an instance of '%s'" % cls)
            registry = cls._myTraitRegistry
        for trait in registry.itervalues():
            descriptor = self._inventory.getTraitDescriptor(trait.name)
            yield Item(trait, descriptor)
        return


    def __iter__(self):
        return self.iteritems()


# end of file
