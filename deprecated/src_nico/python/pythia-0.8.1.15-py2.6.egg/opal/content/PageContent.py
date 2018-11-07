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


class PageContent(PageSection):


    def identify(self, inspector):
        return inspector.onPageContent(self)


    def leftColumn(self, **kwds):
        from PageLeftColumn import PageLeftColumn
        self._leftColumn = PageLeftColumn(**kwds)
        return self._leftColumn


    def main(self, **kwds):
        from PageMain import PageMain
        self._main = PageMain(**kwds)
        return self._main


    def rightColumn(self, **kwds):
        from PageRightColumn import PageRightColumn
        self._rightColumn = PageRightColumn(**kwds)
        return self._rightColumn


    def __init__(self, **kwds):
        PageSection.__init__(self, id="page-header")
        self._leftColumn = None
        self._main = None
        self._rightColumn = None
        return


# version
__id__ = "$Id: PageContent.py,v 1.2 2005/03/22 04:48:46 aivazis Exp $"

# End of file 
