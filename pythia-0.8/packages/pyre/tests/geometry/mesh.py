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

    import journal
    journal.debug("pyre.geometry").activate()

    import pyre.geometry

    mesh = pyre.geometry.mesh(dim=3, order=3)

    return


# main
if __name__ == "__main__":
    test()


# version
__id__ = "$Id: mesh.py,v 1.1.1.1 2005/03/08 16:13:50 aivazis Exp $"

# End of file 
