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


class PageFooter(PageSection):


    def __init__(self):
        PageSection.__init__(self, id="page-footer")
        return


# version
__id__ = "$Id: PageFooter.py,v 1.1 2005/03/20 07:22:58 aivazis Exp $"

# End of file 
