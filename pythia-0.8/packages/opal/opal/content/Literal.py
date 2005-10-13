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


class Literal(object):


    def identify(self, inspector):
        return inspector.onLiteral(self)


    def __init__(self):
        self.text = []
        return


# version
__id__ = "$Id: Literal.py,v 1.1 2005/03/20 07:22:58 aivazis Exp $"

# End of file 
