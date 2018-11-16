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

from pyre.xml.Node import Node


class AbstractNode(Node):



    def _parse(self, expr):
        return self._parser.parse(expr)



    from pyre.units import parser
    _parser = parser()


# version
__id__ = "$Id: AbstractNode.py,v 1.1.1.1 2005/03/08 16:13:45 aivazis Exp $"

#
# End of file
