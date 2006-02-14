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


class Sources(FileContainer):


    def identify(self, inspector):
        return inspector.onSources(self)


# version
__id__ = "$Id: Sources.py,v 1.1.1.1 2005/03/08 16:13:59 aivazis Exp $"

# End of file
