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


class ChainLocator(object):


    def __init__(self, this, next):
        self.this = this
        self.next = next
        return


    def __str__(self):
        return "%s via %s" % (self.this, self.next)


    def __getstate__(self):
        return dict(this = self.this, next = self.next)


    def __setstate__(self, dict):
        self.this = dict['this']
        self.next = dict['next']
        return


    def getAttributes(self, attr):
        return


    __slots__ = ("this", "next")
    

# version
__id__ = "$Id: ChainLocator.py,v 1.1.1.1 2005/03/08 16:13:48 aivazis Exp $"

# End of file 
