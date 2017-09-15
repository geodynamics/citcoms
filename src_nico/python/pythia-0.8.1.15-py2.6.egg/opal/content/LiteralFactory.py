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


class LiteralFactory(object):


    def literal(self):
        from Literal import Literal
        literal = Literal()

        self.contents.append(literal)

        return literal
        

# version
__id__ = "$Id: LiteralFactory.py,v 1.1 2005/03/20 07:22:58 aivazis Exp $"

# End of file 
