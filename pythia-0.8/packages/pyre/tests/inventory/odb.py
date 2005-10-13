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
    import pyre.inventory

    curator = pyre.inventory.curator('test')

    print "curator class:", curator.__class__.__name__
    print "curator.__bases__:", [ c for c in curator.__class__.__mro__]
    print dir(curator)
    
    return


# main
if __name__ == "__main__":
    test()

# version
__id__ = "$Id: odb.py,v 1.1.1.1 2005/03/08 16:13:49 aivazis Exp $"

# End of file 
