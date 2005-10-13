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

from FileContainer import FileContainer


class Headers(FileContainer):


    def identify(self, inspector):
        return inspector.onHeaders(self)


# version
__id__ = "$Id: Headers.py,v 1.1.1.1 2005/03/08 16:13:59 aivazis Exp $"

# End of file
