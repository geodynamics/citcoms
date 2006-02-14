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

import acis
from pyre.geometry.Visitor import Visitor as GeometryVisitor


class Pickler(GeometryVisitor):

    # user callable routine
    def pickle(self, body):
        from Body import Body

        if isinstance(body, Body):
            return body
        
        handle = body.identify(self)

        return Body(handle)


    # solid bodies
    def onBlock(self, block):
        x,y,z = block.diagonal
        return acis.block( (x.value, y.value, z.value) )


    def onCone(self, cone):
        return acis.cone(cone.bottom.value, cone.top.value, cone.height.value)


    def onCylinder(self, cylinder):
        return acis.cylinder(cylinder.radius.value, cylinder.height.value)


    def onPrism(self, prism):
        # NYI
        return self._abstract("onPrism")


    def onPyramid(self, pyramid):
        # NYI
        return self._abstract("onPyramid")


    def onSphere(self, sphere):
        return acis.sphere(sphere.radius.value)


    def onTorus(self, torus):
        return acis.torus(torus.major.value, torus.minor.value)


    def onGeneralizedCone(self, cone):
        return acis.generalizedCone(
            cone.major.value, cone.minor.value, cone.scale, cone.height.value)


    # Euler operations
    def onDifference(self, difference):
        blank = difference.op1.identify(self)
        tool = difference.op2.identify(self)
        return acis.difference(blank, tool)


    def onIntersection(self, intersection):
        blank = intersection.op1.identify(self)
        tool = intersection.op2.identify(self)
        return acis.intersection(blank, tool)


    def onUnion(self, union):
        blank = union.op1.identify(self)
        tool = union.op2.identify(self)
        return acis.union(blank, tool)


    # transformations
    def onDilation(self, dilation):
        body = dilation.body.identify(self)
        return acis.dilation(body, dilation.scale)


    def onReflection(self, reflection):
        body = reflection.body.identify(self)
        return acis.reflection(body, reflection.vector)


    def onReversal(self, reversal):
        body = reversal.body.identify(self)
        return acis.reversal(body)


    def onRotation(self, rotation):
        body = rotation.body.identify(self)
        return acis.rotation(body, rotation.angle, rotation.vector)


    def onTranslation(self, translation):
        body = translation.body.identify(self)
        tx,ty,tz = translation.vector
        return acis.translation(body, (tx.value, ty.value, tz.value))


# version
__id__ = "$Id: Pickler.py,v 1.1.1.1 2005/03/08 16:13:32 aivazis Exp $"

#
# End of file
