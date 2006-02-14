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


class Document(ElementContainer, LiteralFactory, ParagraphFactory):


    def form(self, **kwds):
        from Form import Form
        form = Form(**kwds)
        self.contents.append(form)
        return form


    def identify(self, inspector):
        return inspector.onDocument(self)


    def __init__(self, title, description="", byline="", **kwds):
        ElementContainer.__init__(self, 'div', **kwds)
        LiteralFactory.__init__(self)
        ParagraphFactory.__init__(self)

        self.title = title
        self.description = description
        self.byline = byline
        
        return


# version
__id__ = "$Id: Document.py,v 1.3 2005/03/31 04:39:02 aivazis Exp $"

# End of file 
