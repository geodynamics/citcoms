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


def test():

    from pyre.inventory.Trait import Trait

    trait = Trait(name="trait")

    print "trait = %r" % trait

    t = trait.__get__(None, None)
    print "__get__: %r(%r)" % (t.name, t)

    return


# main
if __name__ == "__main__":
    test()


# version
__id__ = "$Id: trait.py,v 1.1.1.1 2005/03/08 16:13:49 aivazis Exp $"

# End of file 
