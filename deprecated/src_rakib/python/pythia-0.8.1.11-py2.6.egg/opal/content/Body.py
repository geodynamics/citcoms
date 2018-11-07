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
from LiteralFactory import LiteralFactory
from ParagraphFactory import ParagraphFactory


class Body(ElementContainer, LiteralFactory, ParagraphFactory):


    def pageContent(self, **kwds):
        from PageContent import PageContent
        self._content = PageContent(**kwds)
        self.contents.append(self._content)
        return self._content


    def pageCredits(self, **kwds):
        from PageCredits import PageCredits
        credits = PageCredits(**kwds)
        self.contents.append(credits)
        return credits


    def pageHeader(self, **kwds):
        from PageHeader import PageHeader
        header = PageHeader(**kwds)
        self.contents.append(header)
        return header


    def pageFooter(self, **kwds):
        from PageFooter import PageFooter
        footer = PageFooter(**kwds)
        self.contents.append(footer)
        return footer


    def identify(self, inspector):
        return inspector.onBody(self)


    def __init__(self, **kwds):
        ElementContainer.__init__(self, 'body', **kwds)
        LiteralFactory.__init__(self)
        ParagraphFactory.__init__(self)

        # body parts
        self._content = None
        
        return


# version
__id__ = "$Id: Body.py,v 1.3 2005/04/22 03:42:56 pyre Exp $"

# End of file 
