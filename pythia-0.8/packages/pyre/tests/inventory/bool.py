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


from pyre.inventory.Inventory import Inventory


class MyInventory(Inventory):

    import pyre.inventory

    flag = pyre.inventory.bool('flag', default=True)


def test():

    inv1 = MyInventory("inv1")
    print "inv1 = %r" % inv1
    print "    _inventory = %r" % inv1._priv_inventory
    print "    _traitRegistry = %r" % inv1._traitRegistry
    print "    _facilityRegistry = %r" % inv1._facilityRegistry

    inv2 = MyInventory("test")
    print "inv2 = %r" % inv2
    print "    _inventory = %r" % inv2._priv_inventory
    print "    _traitRegistry = %r" % inv2._traitRegistry
    print "    _facilityRegistry = %r" % inv2._facilityRegistry

    print "testing..."
    print "    inv1.flag = %r (True) from %s" % (inv1.flag, inv1.getTraitDescriptor('flag').locator)
    inv1.flag = False
    print "    inv1.flag = %r (False) from %s" % (inv1.flag, inv1.getTraitDescriptor('flag').locator)
    inv1.flag = True
    inv2.flag = "false"
    print "    inv1.flag = %r (True) from %s" % (inv1.flag, inv1.getTraitDescriptor('flag').locator)
    print "    inv2.flag = %r (False) from %s" % (inv2.flag, inv2.getTraitDescriptor('flag').locator)
    inv2.flag = "False"
    print "    inv2.flag = %r (False) from %s" % (inv2.flag, inv2.getTraitDescriptor('flag').locator)
    inv2.flag = "0"
    print "    inv2.flag = %r (False) from %s" % (inv2.flag, inv2.getTraitDescriptor('flag').locator)
    inv2.flag = "no"
    print "    inv2.flag = %r (False) from %s" % (inv2.flag, inv2.getTraitDescriptor('flag').locator)
    inv2.flag = "NO"
    print "    inv2.flag = %r (False) from %s" % (inv2.flag, inv2.getTraitDescriptor('flag').locator)
    inv2.flag = "true"
    print "    inv2.flag = %r (True) from %s" % (inv2.flag, inv2.getTraitDescriptor('flag').locator)
    inv2.flag = "True"
    print "    inv2.flag = %r (True) from %s" % (inv2.flag, inv2.getTraitDescriptor('flag').locator)
    inv2.flag = "yes"
    print "    inv2.flag = %r (True) from %s" % (inv2.flag, inv2.getTraitDescriptor('flag').locator)
    inv2.flag = "Yes"
    print "    inv2.flag = %r (True) from %s" % (inv2.flag, inv2.getTraitDescriptor('flag').locator)
    inv2.flag = "1"
    print "    inv2.flag = %r (True) from %s" % (inv2.flag, inv2.getTraitDescriptor('flag').locator)
    print "done testing"

    return

# main
if __name__ == '__main__':
    test()


# version
__id__ = "$Id: bool.py,v 1.1.1.1 2005/03/08 16:13:49 aivazis Exp $"

# End of file 
