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


from Element import Element


class Link(Element):


    def identify(self, inspector):
        return inspector.onLink(self)


    def __init__(self, **kwds):
        Element.__init__(self, 'link', **kwds)
        return

# version
__id__ = "$Id: Link.py,v 1.1 2005/03/20 07:22:58 aivazis Exp $"

# End of file 
