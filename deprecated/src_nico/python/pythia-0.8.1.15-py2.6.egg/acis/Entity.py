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


class Entity(object):


    def handle(self):
        return self._handle


    def boundingBox(self):
        return acis.box(self._handle)


    def faces(self):
        from Face import Face
        faceHandles = acis.faces(self._handle)

        return tuple([ Face(handle) for handle in faceHandles ])


    def distance(self, other):
        return acis.distance(self._handle, other._handle)


    def touches(self, other):
        return acis.touch(self._handle, other._handle)


    def save(self, file, mode=1):
        acis.save(file, mode, [self._handle])
        return


    # attributes

    def integerAttribute(self, name, value):
        acis.setAttributeInt(self._handle, name, value)
        return


    def realAttribute(self, name, value):
        acis.setAttributeDouble(self._handle, name, value)
        return


    def stringAttribute(self, name, value):
        acis.setAttributeString(self._handle, name, value)
        return


    # meta methods

    def __init__(self, handle):
        self._handle = handle
        return


# version
__id__ = "$Id: Entity.py,v 1.1.1.1 2005/03/08 16:13:32 aivazis Exp $"

#
# End of file
