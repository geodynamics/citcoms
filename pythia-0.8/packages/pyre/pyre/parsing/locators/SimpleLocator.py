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


class SimpleLocator(object):


    def __init__(self, source):
        self.source = source
        return


    def __str__(self):
        return "{%s}" % self.source


    __slots__ = ("source")
    

# version
__id__ = "$Id: SimpleLocator.py,v 1.1.1.1 2005/03/08 16:13:48 aivazis Exp $"

# End of file 
