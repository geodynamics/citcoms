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


class ContentMill(object):


    def onLiteral(self, literal):
        return [ "\n".join(literal.text) ]


    def onParagraph(self, p):
        text = [ self.tagger.onElementBegin(p) ]
        text += p.text
        text.append(self.tagger.onElementEnd(p))
        return text


# version
__id__ = "$Id: ContentMill.py,v 1.1 2005/03/20 07:22:58 aivazis Exp $"

# End of file 
