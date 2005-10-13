#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005 All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


class DummyCommunicator:


    def handle(self):
        return None


    def __init__(self, *args):
        self.rank = 0
        self.size = 0
        return


# version
__id__ = "$Id: DummyCommunicator.py,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $"

# End of file
