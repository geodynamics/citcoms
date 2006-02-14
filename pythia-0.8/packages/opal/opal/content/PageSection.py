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


from ElementContainer import ElementContainer


class PageSection(ElementContainer):


    def identify(self, inspector):
        return inspector.onPageSection(self)


    def __init__(self, **kwds):
        ElementContainer.__init__(self, 'div', **kwds)
        return


# version
__id__ = "$Id: PageSection.py,v 1.1 2005/03/20 07:22:58 aivazis Exp $"

# End of file 
