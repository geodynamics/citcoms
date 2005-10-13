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


class CommunicatorGroup:


    def handle(self):
        return self._handle


    def include(self, included):
        import _mpi
        handle = _mpi.groupInclude(self._handle, included)
        if handle:
            return CommunicatorGroup(handle)
        return None


    def exclude(self, excluded):
        import _mpi
        handle = _mpi.groupExclude(self._handle, excluded)
        if handle:
            return CommunicatorGroup(handle)

        return None


    def __init__(self, handle):
        import _mpi

        self._handle = handle
        self.rank = _mpi.groupRank(handle)
        self.size = _mpi.groupSize(handle)
        return


# version
__id__ = "$Id: CommunicatorGroup.py,v 1.1.1.1 2005/03/08 16:13:29 aivazis Exp $"

# End of file
