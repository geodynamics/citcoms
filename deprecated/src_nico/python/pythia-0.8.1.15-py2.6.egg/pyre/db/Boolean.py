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


from Column import Column


class Boolean(Column):


    def type(self):
        return "boolean" % self.length


    def __init__(self, name, length, **kwds):
        Column.__init__(self, name, **kwds)
        self.length = length
        return


# version
__id__ = "$Id: Boolean.py,v 1.2 2005/04/06 20:56:13 aivazis Exp $"

# End of file 
