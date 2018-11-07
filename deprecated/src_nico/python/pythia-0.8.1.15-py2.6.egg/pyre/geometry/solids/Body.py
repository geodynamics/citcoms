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

class Body(object):


    def identify(self, inspector):
        raise NotImplementedError(
            "class '%s' should override method '%s'" % (self.__class__.__name__, method))



# version
__id__ = "$Id: Body.py,v 1.1.1.1 2005/03/08 16:13:46 aivazis Exp $"

#
# End of file
