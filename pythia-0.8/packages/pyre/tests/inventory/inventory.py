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


from pyre.inventory.Trait import Trait
from pyre.inventory.Property import Property
from pyre.inventory.Inventory import Inventory


class MyInventory(Inventory):

    name = Trait("name")
    prop = Property("prop", "str", default="hello")
    

def test():

    inv = MyInventory("inv")

    print "inventory = %r" % inv
    print "    _inventory = %r" % inv._priv_inventory
    # print "    name = %r (uninitialized)" % inv.name

    print
    print "testing generic traits"
    inv.name = "test"
    print "    name = %r (test)" % inv.name
    print "    __dict__ = %r" % inv.__dict__

    print "class = %r" % type(inv).name

    d = inv._getTraitDescriptor("name")
    print "descriptor = %r" % d
    print "value = %r" % d.value
    print "locator = %s" % d.locator

    print
    print "testing generic properties"
    inv.prop = "property"
    print "    prop = %r (property)" % inv.prop
    print "    __dict__ = %r" % inv.__dict__

    print "class = %r" % type(inv).name

    d = inv._getTraitDescriptor("prop")
    print "descriptor = %r" % d
    print "value = %r" % d.value
    print "locator = %s" % d.locator

    return


# main
if __name__ == "__main__":
    test()


# version
__id__ = "$Id: inventory.py,v 1.1.1.1 2005/03/08 16:13:49 aivazis Exp $"

# End of file 
