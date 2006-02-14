#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                              Michael A.G. Aivazis
#                       California Institute of Technology
#                       (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from Asset import Asset


class Header(Asset):


    def identify(self, inspector):
        return inspector.onHeader(self)


# version
__id__ = "$Id: Header.py,v 1.1.1.1 2005/03/08 16:13:59 aivazis Exp $"

# End of file
