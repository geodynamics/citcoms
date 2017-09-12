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


class Boundary(object):


    def setMesh(self, mesh):
        self.mesh = mesh
        dim, order, vertices, triangles = mesh.statistics()

        import elc
        self.velocity = elc.allocateField(3, vertices)
        self.pressure = elc.allocateField(1, vertices)

        return
        

    def __init__(self):
        self.mesh = None
        self.velocity = None
        self.pressure = None

        return


# version
__id__ = "$Id: Boundary.py,v 1.1.1.1 2005/03/08 16:13:28 aivazis Exp $"

# End of file 
