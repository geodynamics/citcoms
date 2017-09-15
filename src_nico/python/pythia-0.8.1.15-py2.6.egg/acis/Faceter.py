#!/usr/bin/env python
#
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005 All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from pyre.components.Component import Component


class Faceter(Component):


    class Inventory(Component.Inventory):

        import pyre.inventory

        gridAspectRatio = pyre.inventory.float("gridAspectRatio")
        maximumEdgeLength = pyre.inventory.float("maximumEdgeLength")
        maximumSurfaceTolerance = pyre.inventory.float("maximumSurfaceTolerance")


    def facet(self, body):

        from Pickler import Pickler
        pickler = Pickler()
        body = pickler.pickle(body)
        
        import acis
        import pyre.geometry

        mesh = pyre.geometry.mesh(3,3)
        properties = self.inventory
        acis.facet(mesh.handle(), body.handle(), properties)
        bbox = acis.box(body.handle())

        return mesh, bbox


    def mesh(self, body):
        from Pickler import Pickler
        pickler = Pickler()
        body = pickler.pickle(body)
        
        import acis

        properties = self.inventory
        meshed = acis.mesh(body.handle(), properties)

        from Body import Body
        return Body(meshed)


    def __init__(self, options=None):
        Component.__init__(self, "acis-faceter", "surfaceMesher")
        return


# version
__id__ = "$Id: Faceter.py,v 1.1.1.1 2005/03/08 16:13:32 aivazis Exp $"

#
# End of file
