#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                      California Institute of Technology
#                        (C) 2008  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from pyre.inventory.Facility import Facility


class FacilityArrayFacility(Facility):


    def __init__(self, name, itemFactory, **kwds):
        Facility.__init__(self, name=name, **kwds)
        self.itemFactory = itemFactory
        return


    def _retrieveComponent(self, instance, componentName):
        facilityNames = self._cast(componentName)
        
        dict = {}
        for index, facilityName in enumerate(facilityNames):
            facility = self.itemFactory(facilityName)
            attr = "item%d" % index
            dict[attr] = facility

        from Inventory import Inventory
        from pyre.components.Component import Component
        
        Inventory = Inventory.__metaclass__("FacilityArray.Inventory", (Component.Inventory,), dict)

        dict = {'Inventory': Inventory}
        FacilityArray = Component.__metaclass__("FacilityArray", (Component,), dict)
        fa = FacilityArray(self.name)

        import pyre.parsing.locators
        locator = pyre.parsing.locators.builtIn()

        return fa, locator


    def _cast(self, text):
        if isinstance(text, basestring):
            if text and text[0] in '[({':
                text = text[1:]
            if text and text[-1] in '])}':
                text = text[:-1]
                
            value = text.split(",")

            # allow trailing comma
            if len(value) and not value[-1]:
                value.pop()
        else:
            value = text

        if isinstance(value, list):
            return value
            
        raise TypeError("facility '%s': could not convert '%s' to a list" % (self.name, text))



# end of file
