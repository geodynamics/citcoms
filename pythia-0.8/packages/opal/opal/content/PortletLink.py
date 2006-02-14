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


class PortletLink(Element):


    def identify(self, inspector):
        return inspector.onPortletLink(self)


    def __init__(self, description, type='', target='', tip='', icon='', **kwds):
        Element.__init__(self, 'a', **kwds)

        self.description = description
        self.type = type
        self.target = target
        self.tip = tip
        self.icon = icon

        return


# version
__id__ = "$Id: PortletLink.py,v 1.1 2005/05/05 04:45:08 pyre Exp $"

# End of file 
