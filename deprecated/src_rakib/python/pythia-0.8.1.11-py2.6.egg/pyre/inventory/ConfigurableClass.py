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


class ConfigurableClass(type):

    
    class TraitForwarder(object):
        def __init__(self, trait):
            self.trait = trait
        def __get__(self, instance, cls=None):
            return self.trait.__get__(instance and instance.inventory or None,
                                      cls and cls.Inventory or None)
        def __set__(self, instance, value):
            self.trait.__set__(instance.inventory, value)


    def __init__(cls, name, bases, dct):
        
        type.__init__(cls, name, bases, dct)
        
        if dct.has_key('Inventory'):
            # allow for traditional components
            return
        
        from Notary import Notary
        from Trait import Trait
        
        # derive the bases for the inventory class
        inventoryBases = []
        bases = list(bases)
        for base in bases:
            baseInventory = getattr(base, 'Inventory', None)
            if baseInventory:
                inventoryBases.append(baseInventory)
        inventoryBases = tuple(inventoryBases)

        # populate the inventory class dictionary
        import pyre.inventory
        inventoryDict = {}
        for traitName, trait in [kv for kv in dct.iteritems() if isinstance(kv[1], Trait)]:
            # move the trait to the inventory...
            inventoryDict[traitName] = trait
            # ...replacing it with a forwarder
            setattr(cls, traitName, ConfigurableClass.TraitForwarder(trait))

        # create the inventory class
        cls.Inventory = Notary('%s.Inventory' % name,
                               inventoryBases,
                               inventoryDict)


# end of file 
