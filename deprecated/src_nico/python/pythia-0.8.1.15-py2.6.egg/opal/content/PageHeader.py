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


class PageHeader(PageSection):


    def logo(self, **kwds):
        from Logo import Logo
        logo = Logo(**kwds)
        self.contents.append(logo)
        return logo


    def personalTools(self, **kwds):
        from PersonalTools import PersonalTools
        tools = PersonalTools(**kwds)
        self.contents.append(tools)
        return tools


    def searchBox(self, **kwds):
        from SearchBox import SearchBox
        searchBox = SearchBox(**kwds)
        self.contents.append(searchBox)
        return searchBox


    def __init__(self):
        PageSection.__init__(self, id="page-header")
        return


# version
__id__ = "$Id: PageHeader.py,v 1.1 2005/03/20 07:22:58 aivazis Exp $"

# End of file 
