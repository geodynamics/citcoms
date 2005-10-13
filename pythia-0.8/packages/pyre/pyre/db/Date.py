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


class Date(Column):


    def type(self):
        return "date"


    def __init__(self, name, **kwds):
        Column.__init__(self, name, **kwds)
        return


# version
__id__ = "$Id: Date.py,v 1.1 2005/04/06 20:46:30 aivazis Exp $"

# End of file 
