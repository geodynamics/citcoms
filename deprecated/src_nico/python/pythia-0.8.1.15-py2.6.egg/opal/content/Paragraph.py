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


class Paragraph(Element):


    def identify(self, inspector):
        return inspector.onParagraph(self)


    def __init__(self, **kwds):
        Element.__init__(self, 'p', **kwds)

        self.text = []
        return


# version
__id__ = "$Id: Paragraph.py,v 1.1 2005/03/20 07:22:58 aivazis Exp $"

# End of file 
