#!/usr/bin/env python
#
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005 All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from Primitive import Primitive


class Block(Primitive):


    def identify(self, visitor):
        return visitor.onBlock(self)


    def __init__(self, diagonal):
        self.diagonal = tuple(diagonal)

        self._info.log("new %s" % self)
                 
        return


    def __str__(self):
        return "block: diagonal=(%s, %s, %s)" % self.diagonal


# version
__id__ = "$Id: Block.py,v 1.1.1.1 2005/03/08 16:13:46 aivazis Exp $"

#
# End of file
