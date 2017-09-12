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

from Composition import Composition


class Transformation(Composition):


    def __init__(self, attributes):
        Composition.__init__(self, attributes)
        self._body = None
        return


    def _setOperand(self, body):
        self._body = body
        return


# version
__id__ = "$Id: Transformation.py,v 1.1.1.1 2005/03/08 16:13:45 aivazis Exp $"

# End of file
