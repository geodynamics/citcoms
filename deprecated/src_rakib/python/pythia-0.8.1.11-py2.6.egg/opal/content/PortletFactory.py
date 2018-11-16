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


class PortletFactory(object):


    def portlet(self, **kwds):
        from Portlet import Portlet
        portlet = Portlet(**kwds)

        self.contents.append(portlet)

        return portlet


# version
__id__ = "$Id: PortletFactory.py,v 1.1 2005/03/26 01:50:10 aivazis Exp $"

# End of file 
