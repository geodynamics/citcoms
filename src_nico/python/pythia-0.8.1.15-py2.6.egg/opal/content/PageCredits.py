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
from ParagraphFactory import ParagraphFactory


class PageCredits(PageSection, ParagraphFactory):


    def __init__(self):
        PageSection.__init__(self, id="page-credits")
        ParagraphFactory.__init__(self)
        return


# version
__id__ = "$Id: PageCredits.py,v 1.1 2005/03/20 07:22:58 aivazis Exp $"

# End of file 
