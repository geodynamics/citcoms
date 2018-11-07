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


class ParagraphFactory(object):


    def paragraph(self, **kwds):
        from Paragraph import Paragraph
        paragraph = Paragraph(**kwds)

        self.contents.append(paragraph)

        return paragraph


# version
__id__ = "$Id: ParagraphFactory.py,v 1.1 2005/03/20 07:22:58 aivazis Exp $"

# End of file 
