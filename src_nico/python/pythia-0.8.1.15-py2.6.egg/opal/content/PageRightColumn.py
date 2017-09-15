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


from PageSection import PageSection
from PortletFactory import PortletFactory


class PageRightColumn(PageSection, PortletFactory):


    def __init__(self, **kwds):
        PageSection.__init__(self, **kwds)
        PortletFactory.__init__(self)
        return


# version
__id__ = "$Id: PageRightColumn.py,v 1.3 2005/03/27 01:16:06 aivazis Exp $"

# End of file 
