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


from pyre.inventory.Facility import Facility
from pyre.inventory.Inventory import Inventory


class MyFacility(Facility):


    def entry1(self):
        """abstract test function"""


    def entry2(self):
        """abstract test function"""


class MyInventory(Inventory):

    import pyre.inventory

    facility = pyre.inventory.facility("facility")
    

def test():

    print "facility Facility"
    print "    _interfaceRegistry = %r" % Facility._interfaceRegistry
    print "facility MyFacility"
    print "    _interfaceRegistry = %r" % MyFacility._interfaceRegistry

    inv = MyInventory("inv")

    print "inventory = %r" % inv
    print "    _inventory = %r" % inv._priv_inventory

    return


# main
if __name__ == "__main__":
    test()


# version
__id__ = "$Id: component.py,v 1.1.1.1 2005/03/08 16:13:49 aivazis Exp $"

# End of file 
