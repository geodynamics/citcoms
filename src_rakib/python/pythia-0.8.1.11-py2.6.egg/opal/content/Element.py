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


class Element(object):


    def identify(self, inspector):
        raise NotImplementedError("class %r should implement 'identify'" % self.__class__.__name__)


    def __init__(self, tag, **attributes):
        self.tag = tag
        self.attributes = attributes
        return


# version
__id__ = "$Id: Element.py,v 1.2 2005/03/27 10:17:02 aivazis Exp $"

# End of file 
