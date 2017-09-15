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


class Time(Column):


    def type(self):
        if not self.tz:
            return "time without time zone"
        return "time"


    def __init__(self, name, tz=True, **kwds):
        Column.__init__(self, name, **kwds)
        self.tz = tz
        return


# version
__id__ = "$Id: Time.py,v 1.1 2005/04/06 20:46:30 aivazis Exp $"

# End of file 
